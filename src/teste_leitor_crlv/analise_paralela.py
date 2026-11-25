from __future__ import annotations

import base64
import binascii
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .analise_local import validar_local_com_checker
from .analise_ocr import analisar_crlv_com_posicao
from .analise_paddleocr import analisar_crlv_com_paddleocr
from .qrcode_utils import extrair_bytes_qrcode, extrair_placa_de_dicionario

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Análise Rápida Paralela (Corrida Multithread)"])

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="ocr_race")


class AnaliseRapidaRequest(BaseModel):
    imagem_base64: str | None = Field(
        default=None,
        description="Imagem em Base64 ou Data URI (PDF ou Foto)",
    )
    caminho_arquivo: str | None = Field(
        default=None,
        description="Caminho absoluto ou relativo do arquivo local",
    )
    garantir_treino_paddle: bool = Field(
        default=True,
        description="Mantém processamento do PaddleOCR em segundo plano para alimentar dataset de falhas/treino",
    )
    margem_erro: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Tolerância geométrica das âncoras (10% padrão)",
    )


def _executar_digital_checker(imagem_bytes: bytes) -> dict[str, Any]:
    """Executa DigitalDocumentChecker (offline, criptografia do QR Code)."""
    t0 = time.perf_counter()
    try:
        dados_binarios = extrair_bytes_qrcode(imagem_bytes)
        b64_qr = base64.b64encode(dados_binarios).decode("ascii")
        resultado_dict = validar_local_com_checker(b64_qr, include_raw=False)
        sucesso = bool(resultado_dict.get("status", {}).get("is_parsed", False))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "sucesso": sucesso,
            "analisador": "digital_document_checker",
            "tempo_ms": elapsed_ms,
            "dados": resultado_dict,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "sucesso": False,
            "analisador": "digital_document_checker",
            "tempo_ms": elapsed_ms,
            "erro": str(exc),
        }


def _executar_easyocr(imagem_bytes: bytes, margem_erro: float) -> dict[str, Any]:
    """Executa EasyOCR com análise geométrica."""
    t0 = time.perf_counter()
    try:
        resultado = analisar_crlv_com_posicao(
            origem=imagem_bytes,
            margem_erro=margem_erro,
            otimizar_enquadramento=True,
        )
        sucesso = bool(resultado.get("formato_crlv_valido", False))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "sucesso": sucesso,
            "analisador": "easyocr",
            "tempo_ms": elapsed_ms,
            "dados": resultado,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "sucesso": False,
            "analisador": "easyocr",
            "tempo_ms": elapsed_ms,
            "erro": str(exc),
        }


def _executar_paddleocr(
    imagem_bytes: bytes,
    margem_erro: float,
    nome_arquivo: str = "",
    salvar_amostra: bool = False,
) -> dict[str, Any]:
    """Executa PaddleOCR (com modelo local e salvamento para retreino)."""
    t0 = time.perf_counter()
    try:
        resultado = analisar_crlv_com_paddleocr(
            origem=imagem_bytes,
            margem_erro=margem_erro,
            nome_arquivo=nome_arquivo,
            salvar_amostra=salvar_amostra,
            otimizar_enquadramento=True,
        )
        sucesso = bool(resultado.get("formato_crlv_valido", False))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "sucesso": sucesso,
            "analisador": "paddleocr",
            "tempo_ms": elapsed_ms,
            "dados": resultado,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "sucesso": False,
            "analisador": "paddleocr",
            "tempo_ms": elapsed_ms,
            "erro": str(exc),
        }


def orquestrar_corrida_paralela(
    imagem_bytes: bytes,
    margem_erro: float = 0.10,
    nome_arquivo: str = "",
    garantir_treino_paddle: bool = True,
    background_tasks: BackgroundTasks | None = None,
) -> dict[str, Any]:
    """Orquestra corrida inteligente: Fast-Path com DigitalDocumentChecker (<50ms).

    Se o QR Code for autêntico, encerra de imediato economizando CPU.
    Caso contrário, aciona OCRs em paralelo com prioridade para PaddleOCR.
    """
    t_inicio = time.perf_counter()

    # 1. Fast-Path Ultrarrápido: Checa assinatura criptográfica do QR Code primeiro (~30-50ms)
    res_checker = _executar_digital_checker(imagem_bytes)
    if res_checker.get("sucesso") is True:
        tempo_total = round((time.perf_counter() - t_inicio) * 1000, 2)
        paddle_em_background = False

        # Se solicitado, submete o PaddleOCR de forma assíncrona para registrar amostras de treino
        if garantir_treino_paddle:
            _executor.submit(
                _executar_paddleocr,
                imagem_bytes,
                margem_erro,
                nome_arquivo,
                True,
            )
            paddle_em_background = True

        placa = extrair_placa_de_dicionario(res_checker["dados"])
        return {
            "sucesso": True,
            "validador": "digital_document_checker",
            "arquivo": nome_arquivo or None,
            "placa": placa,
            "payload": {
                "fast_path": True,
                "tempo_resposta_ms": tempo_total,
                "paddle_coletando_em_background": paddle_em_background,
                "dados": res_checker["dados"],
            },
        }

    # 2. Se o QR Code falhou ou não existe, dispara OCRs em paralelo
    fut_paddle = _executor.submit(
        _executar_paddleocr,
        imagem_bytes,
        margem_erro,
        nome_arquivo,
        garantir_treino_paddle,
    )
    fut_easy = _executor.submit(_executar_easyocr, imagem_bytes, margem_erro)

    mapeamento_futuros = {
        fut_paddle: "paddleocr",
        fut_easy: "easyocr",
    }

    primeiro_sucesso: dict[str, Any] | None = None
    resultados_concluidos: dict[str, Any] = {"digital_document_checker": res_checker}

    for fut in concurrent.futures.as_completed(mapeamento_futuros):
        nome_analisador = mapeamento_futuros[fut]
        try:
            res = fut.result()
            resultados_concluidos[nome_analisador] = res
            if res.get("sucesso") is True and primeiro_sucesso is None:
                primeiro_sucesso = res
                break
        except Exception as exc:  # noqa: BLE001
            resultados_concluidos[nome_analisador] = {"sucesso": False, "erro": str(exc)}

    # Se obtivemos sucesso imediato no OCR:
    if primeiro_sucesso is not None:
        tempo_total = round((time.perf_counter() - t_inicio) * 1000, 2)
        vencedor = primeiro_sucesso["analisador"]

        paddle_em_background = False
        if vencedor != "paddleocr" and garantir_treino_paddle:
            paddle_em_background = True

        placa = extrair_placa_de_dicionario(primeiro_sucesso["dados"])
        payload = {
            "fast_path": False,
            "tempo_resposta_ms": tempo_total,
            "paddle_coletando_em_background": paddle_em_background,
            "dados": primeiro_sucesso["dados"],
        }
        return {
            "sucesso": True,
            "validador": vencedor,
            "arquivo": nome_arquivo or None,
            "placa": placa,
            "payload": payload,
        }

    # Se nenhum venceu, aguarda o outro concluir
    for fut, nome in mapeamento_futuros.items():
        if nome not in resultados_concluidos:
            try:
                resultados_concluidos[nome] = fut.result()
            except Exception as exc:  # noqa: BLE001
                resultados_concluidos[nome] = {"sucesso": False, "erro": str(exc)}

    tempo_total = round((time.perf_counter() - t_inicio) * 1000, 2)
    payload_falha = {
        "mensagem": "Nenhum analisador obteve sucesso na leitura do documento.",
        "tempo_total_ms": tempo_total,
        "tentativas": {
            k: {"sucesso": v.get("sucesso", False), "tempo_ms": v.get("tempo_ms"), "erro": v.get("erro")}
            for k, v in resultados_concluidos.items()
        },
    }
    return {
        "sucesso": False,
        "validador": "nenhum",
        "arquivo": nome_arquivo or None,
        "placa": None,
        "payload": payload_falha,
    }


@router.post("/api/analisar-rapido-imagem")
async def analisar_rapido_upload(
    file: Annotated[UploadFile, File(description="Arquivo de CRLV (PDF ou Imagem)")],
    garantir_treino_paddle: Annotated[bool, Form()] = True,
    margem_erro: Annotated[float, Form()] = 0.10,
    background_tasks: BackgroundTasks = None,
):
    """Envia a imagem aos 3 analisadores em paralelo e retorna o primeiro com sucesso."""
    conteudo_arquivo = await file.read()
    if not conteudo_arquivo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    resultado = orquestrar_corrida_paralela(
        imagem_bytes=conteudo_arquivo,
        margem_erro=margem_erro,
        nome_arquivo=file.filename or "",
        garantir_treino_paddle=garantir_treino_paddle,
        background_tasks=background_tasks,
    )
    return resultado


@router.post("/api/analisar-rapido")
def analisar_rapido_json(
    req: AnaliseRapidaRequest,
    background_tasks: BackgroundTasks = None,
):
    """Envia o documento em JSON aos 3 analisadores em paralelo e retorna o primeiro com sucesso."""
    if req.caminho_arquivo:
        path = Path(req.caminho_arquivo)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {path}")
        try:
            conteudo_bytes = path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo: {exc}") from exc
        nome_arquivo = path.name
    elif req.imagem_base64:
        b64 = req.imagem_base64
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            conteudo_bytes = base64.b64decode(b64)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Base64 inválido: {exc}") from exc
        nome_arquivo = "base64_upload.png"
    else:
        raise HTTPException(
            status_code=400,
            detail="Informe imagem_base64 ou caminho_arquivo.",
        )

    resultado = orquestrar_corrida_paralela(
        imagem_bytes=conteudo_bytes,
        margem_erro=req.margem_erro,
        nome_arquivo=nome_arquivo,
        garantir_treino_paddle=req.garantir_treino_paddle,
        background_tasks=background_tasks,
    )
    return resultado
