import base64
import binascii
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from .analise_ocr import (
    carregar_imagem,
    corrigir_placa,
    delimitar_documento_pelos_textos,
    dentro_da_area,
)
from .dispositivo import detectar_providers_onnx
from .enquadramento import otimizar_enquadramento_crlv
from .reconhecedor_local import ReconhecedorCRNNLocal, obter_reconhecedor_local

router = APIRouter(tags=["Leitura OCR com PaddleOCR (CRLV)"])

logger = logging.getLogger(__name__)

_paddle_ocr_instance = None

DIRETORIO_TREINAMENTO = Path("dados_treinamento")
DIRETORIO_FALHAS = DIRETORIO_TREINAMENTO / "falhas"
DIRETORIO_SUCESSOS = DIRETORIO_TREINAMENTO / "sucessos"


def obter_reader_paddle(model_dir: str | None = None):
    """Retorna instância singleton do PaddleOCR (pt, engine onnxruntime)."""
    global _paddle_ocr_instance
    caminho_custom = model_dir or os.getenv("PADDLE_MODEL_DIR")

    if _paddle_ocr_instance is None or model_dir is not None:
        from paddleocr import PaddleOCR

        kwargs = {"lang": "pt", "engine": "onnxruntime"}
        if caminho_custom and Path(caminho_custom).exists():
            kwargs["rec_model_dir"] = str(caminho_custom)

        instance = PaddleOCR(**kwargs)
        if model_dir is None:
            _paddle_ocr_instance = instance
        return instance
    return _paddle_ocr_instance


class OcrPaddleRequest(BaseModel):
    caminho_arquivo: str | None = Field(
        default=None,
        description="Caminho do arquivo local de imagem ou PDF do CRLV",
    )
    imagem_base64: str | None = Field(
        default=None,
        description="Arquivo de imagem ou PDF em Base64 ou Data URI",
    )
    arquivo_base64: str | None = Field(
        default=None,
        description="Alias para imagem_base64",
    )
    margem_erro: float = Field(
        default=0.10,
        description="Margem de tolerância para posicionamento das âncoras (default 0.10)",
    )
    salvar_amostra: bool = Field(
        default=False,
        description="Salva cópia e metadados na pasta dados_treinamento/ para active learning",
    )
    usar_modelo_local: bool = Field(
        default=True,
        description="Prioriza modelo ONNX treinado localmente se disponível em treinamento/modelos/",
    )


class FeedbackAmostraRequest(BaseModel):
    id_amostra: str = Field(..., description="ID da amostra retornada pelo OCR")
    placa_correta: str | None = Field(None, description="Placa real corrigida")
    renavam_correto: str | None = Field(None, description="RENAVAM real corrigido")
    observacoes: str | None = Field(None, description="Observações adicionais para dataset")


def salvar_amostra_treino(
    imagem: np.ndarray,
    analise: dict,
    nome_origem: str = "",
) -> str | None:
    """Salva imagem e anotações para compor dataset de retreino."""
    try:
        valido = analise.get("formato_crlv_valido", False)
        pasta = DIRETORIO_SUCESSOS if valido else DIRETORIO_FALHAS
        pasta.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        identificador = uuid.uuid4().hex[:8]
        prefixo = f"{timestamp}_{identificador}"

        caminho_img = pasta / f"{prefixo}.jpg"
        caminho_json = pasta / f"{prefixo}.json"

        Image.fromarray(imagem).save(caminho_img, quality=95)

        dados_salvos = {
            "id_amostra": prefixo,
            "data": timestamp,
            "nome_origem": nome_origem,
            "valido": valido,
            "analise": analise,
        }
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(dados_salvos, f, ensure_ascii=False, indent=2)

        return prefixo
    except (OSError, ValueError) as exc:
        logger.warning("Falha ao salvar amostra de treino: %s", exc)
        return None


def extrair_resultados_paddle(paddle_res) -> list:
    """Converte a saída do PaddleOCR para formato padrão [(bbox, texto, confianca)]."""
    resultados = []
    if not paddle_res or len(paddle_res) == 0:
        return resultados

    primeira_pagina = paddle_res[0]

    if isinstance(primeira_pagina, dict):
        polys = primeira_pagina.get("rec_polys", [])
        texts = primeira_pagina.get("rec_texts", [])
        scores = primeira_pagina.get("rec_scores", [])
        for poly, text, score in zip(polys, texts, scores):
            bbox = poly.tolist() if hasattr(poly, "tolist") else list(poly)
            resultados.append((bbox, str(text), float(score)))

    elif isinstance(primeira_pagina, (list, tuple)):
        for item in primeira_pagina:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                bbox = item[0]
                text_info = item[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    resultados.append((bbox, str(text_info[0]), float(text_info[1])))
                else:
                    resultados.append((bbox, str(text_info), 1.0))

    return resultados


def analisar_crlv_com_paddleocr(
    origem: str | Path | bytes | np.ndarray,
    margem_erro: float = 0.10,
    nome_arquivo: str = "",
    reader=None,
    salvar_amostra: bool = False,
    usar_modelo_local: bool = True,
    otimizar_enquadramento: bool = True,
) -> dict:
    if isinstance(origem, np.ndarray):
        imagem = origem
    else:
        imagem = carregar_imagem(origem, nome_arquivo=nome_arquivo)

    meta_enquadramento = None
    if otimizar_enquadramento:
        imagem, meta = otimizar_enquadramento_crlv(imagem)
        meta_enquadramento = meta.to_dict()

    if reader is None:
        reader = obter_reader_paddle()

    # Executa detecção/leitura inicial no PaddleOCR
    paddle_out = reader.predict(imagem)
    resultados = extrair_resultados_paddle(paddle_out)

    # Identifica se o modelo treinado localmente está ativo
    reconhecedor_local = (
        obter_reconhecedor_local() if (usar_modelo_local and ReconhecedorCRNNLocal.modelo_disponivel()) else None
    )
    modelo_utilizado = (
        f"local_treinado ({reconhecedor_local.modelo_path.name})"
        if reconhecedor_local
        else "paddleocr_base (onnxruntime)"
    )

    # 1. Delimita área real do documento
    doc_x_min, doc_y_min, doc_x_max, doc_y_max = delimitar_documento_pelos_textos(resultados)
    doc_largura = max(doc_x_max - doc_x_min, 1)
    doc_altura = max(doc_y_max - doc_y_min, 1)

    # 2. Zonas relativas (0.0 a 1.0)
    zonas_ancoras = {
        "REPUBLICA FEDERATIVA": (0.10, 0.00, 0.85, 0.15),
        "CERTIFICADO DE REGISTRO": (0.00, 0.18, 0.85, 0.36),
        "RENAVAM": (0.00, 0.35, 0.45, 0.52),
        "PLACA": (0.00, 0.48, 0.35, 0.65),
        "EXERCICIO": (0.25, 0.48, 0.55, 0.65),
        "NUMERO DO CRV": (0.00, 0.72, 0.45, 0.92),
    }

    ancoras_validadas = {chave: False for chave in zonas_ancoras}
    linhas_texto = []

    for bbox, texto, confianca in resultados:
        texto_limpo = (
            texto.upper()
            .replace("Í", "I")
            .replace("Ú", "U")
            .replace("Ã", "A")
            .replace("Ç", "C")
            .replace("É", "E")
            .replace("Ó", "O")
        )
        linhas_texto.append(texto_limpo)

        try:
            x_centro_abs = sum(p[0] for p in bbox) / len(bbox)
            y_centro_abs = sum(p[1] for p in bbox) / len(bbox)
            x_rel = (x_centro_abs - doc_x_min) / doc_largura
            y_rel = (y_centro_abs - doc_y_min) / doc_altura
        except (IndexError, TypeError, ZeroDivisionError):
            continue

        for ancora, limites in zonas_ancoras.items():
            if ancora in texto_limpo and dentro_da_area(x_rel, y_rel, limites, margem=margem_erro):
                ancoras_validadas[ancora] = True

    texto_completo = " ".join(linhas_texto)
    total_validadas = sum(ancoras_validadas.values())
    layout_valido = total_validadas >= 4

    # 3. Placa
    padrao_placa = r"\b([A-Z0-9]{3})[-\s]?([0-9A-Z]{4})\b"
    placa_encontrada = None

    for i, linha in enumerate(linhas_texto):
        if "PLACA" in linha:
            trecho_busca = " ".join(linhas_texto[i : i + 4])
            trecho_busca = trecho_busca.replace("PLACA", "").replace("EXERCICIO", "")
            match = re.search(padrao_placa, trecho_busca)
            if match:
                candidata = corrigir_placa(f"{match.group(1)}{match.group(2)}")
                if re.match(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$", candidata):
                    placa_encontrada = candidata
                    break

    if not placa_encontrada:
        for candidato in re.findall(r"\b[A-Z0-9]{7}\b", texto_completo):
            corrigida = corrigir_placa(candidato)
            if re.match(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$", corrigida):
                placa_encontrada = corrigida
                break

    # 4. RENAVAM (11 dígitos)
    renavam_match = re.search(r"\b\d{11}\b", texto_completo)

    resultado = {
        "motor": modelo_utilizado,
        "acelerador": detectar_providers_onnx()[0],
        "base_treinamento_local_ativa": reconhecedor_local is not None,
        "formato_crlv_valido": layout_valido and (placa_encontrada is not None) and (renavam_match is not None),
        "placa": placa_encontrada,
        "ancoras_na_posicao": ancoras_validadas,
        "total_ancoras_ok": f"{total_validadas}/6",
        "renavam": renavam_match.group(0) if renavam_match else None,
        "enquadramento": meta_enquadramento,
        "area_documento_detectada": {
            "x_min": int(doc_x_min),
            "y_min": int(doc_y_min),
            "x_max": int(doc_x_max),
            "y_max": int(doc_y_max),
        },
    }

    # Salva para active learning se requisitado ou em caso de falha
    auto_salvar_falhas = os.getenv("PADDLE_AUTO_SAVE_FAILURES", "1") == "1"
    deve_salvar = salvar_amostra or (auto_salvar_falhas and not resultado["formato_crlv_valido"])

    if deve_salvar:
        id_salvo = salvar_amostra_treino(imagem, resultado, nome_origem=nome_arquivo)
        if id_salvo:
            resultado["id_amostra_salva"] = id_salvo

    return resultado


@router.post("/api/paddleocr-crlv-imagem")
async def ocr_paddle_upload(
    file: Annotated[UploadFile, File(description="Arquivo de documento CRLV (PDF, PNG, JPG, etc)")],
    margem_erro: Annotated[float, Form()] = 0.10,
    salvar_amostra: Annotated[bool, Form()] = False,
    usar_modelo_local: Annotated[bool, Form()] = True,
):
    """Lê documento CRLV via PaddleOCR (com prioridade para modelo treinado local se existente)."""
    conteudo_arquivo = await file.read()
    if not conteudo_arquivo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    resultado = analisar_crlv_com_paddleocr(
        origem=conteudo_arquivo,
        margem_erro=margem_erro,
        nome_arquivo=file.filename or "",
        salvar_amostra=salvar_amostra,
        usar_modelo_local=usar_modelo_local,
    )
    return {
        "sucesso": bool(resultado.get("formato_crlv_valido", False)),
        "validador": "paddleocr",
        "arquivo": file.filename,
        "placa": resultado.get("placa"),
        "payload": resultado,
    }


@router.post("/api/paddleocr-crlv")
def ocr_paddle_json(req: OcrPaddleRequest):
    """Recebe imagem ou PDF em Base64 ou caminho e analisa CRLV (priorizando modelo treinado local)."""
    if req.caminho_arquivo:
        path = Path(req.caminho_arquivo)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {path}")
        resultado = analisar_crlv_com_paddleocr(
            origem=str(path),
            margem_erro=req.margem_erro,
            salvar_amostra=req.salvar_amostra,
            usar_modelo_local=req.usar_modelo_local,
        )
    elif req.imagem_base64 or req.arquivo_base64:
        b64 = req.imagem_base64 or req.arquivo_base64
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(b64)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Base64 inválido: {exc}") from exc

        resultado = analisar_crlv_com_paddleocr(
            origem=raw_bytes,
            margem_erro=req.margem_erro,
            salvar_amostra=req.salvar_amostra,
            usar_modelo_local=req.usar_modelo_local,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Informe caminho_arquivo, imagem_base64 ou arquivo_base64.",
        )

    nome_arquivo = (
        Path(req.caminho_arquivo).name
        if req.caminho_arquivo
        else ("base64_upload.png" if (req.imagem_base64 or req.arquivo_base64) else None)
    )

    return {
        "sucesso": bool(resultado.get("formato_crlv_valido", False)),
        "validador": "paddleocr",
        "arquivo": nome_arquivo,
        "placa": resultado.get("placa"),
        "payload": resultado,
    }


@router.post("/api/paddleocr-feedback")
def receber_feedback(req: FeedbackAmostraRequest):
    """Registra rótulo correto fornecido pelo usuário para enriquecer o dataset de treino."""
    encontrado = None
    for pasta in [DIRETORIO_FALHAS, DIRETORIO_SUCESSOS]:
        json_path = pasta / f"{req.id_amostra}.json"
        if json_path.exists():
            encontrado = json_path
            break

    if not encontrado:
        raise HTTPException(status_code=404, detail=f"Amostra {req.id_amostra} não encontrada.")

    try:
        with open(encontrado, "r", encoding="utf-8") as f:
            dados = json.load(f)

        dados["feedback_usuario"] = {
            "placa_correta": req.placa_correta,
            "renavam_correto": req.renavam_correto,
            "observacoes": req.observacoes,
            "data_feedback": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        with open(encontrado, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

        return {"sucesso": True, "mensagem": "Feedback gravado com sucesso para retreino."}
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {exc}") from exc
