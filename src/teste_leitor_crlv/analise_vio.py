import base64
import binascii
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from vio_api import (
    VioApiError,
    VioBadRequestError,
    VioClient,
    VioEnvironmentError,
    VioNotFoundError,
    VioPaymentRequiredError,
    VioUnauthorizedError,
)

from .qrcode_utils import extrair_bytes_qrcode, extrair_placa_de_dicionario, gerar_qrcode_base64

router = APIRouter(tags=["Remoto (API VIO Serpro)"])


class ConsultaVioRequest(BaseModel):
    imagem_base64: str | None = Field(
        default=None,
        description="Imagem em Base64 ou Data URI contendo o QR Code VIO",
    )
    caminho_imagem: str | None = Field(
        default=None,
        description="Caminho local do arquivo de imagem contendo o QR Code",
    )
    texto: str | None = Field(
        default=None,
        description="Texto bruto (caso queira gerar direto do texto)",
    )
    wait: bool = Field(default=True, description="Se True, aguarda o processamento na API VIO")
    incluir_qrcode_base64: bool = Field(default=False, description="Inclui imagem PNG em base64 do QR reconstituído")


def consultar_vio_com_base64(b64_qr: str, wait: bool) -> dict:
    """Envia o QR Code reconstruído para o cliente integrador-vio (API Serpro)."""
    try:
        client = VioClient()
        return client.read_qrcode(b64_qr, wait=wait)
    except VioUnauthorizedError as exc:
        raise HTTPException(status_code=401, detail=f"VIO Não autorizado: {exc}") from exc
    except VioPaymentRequiredError as exc:
        raise HTTPException(status_code=402, detail=f"VIO Saldo insuficiente: {exc}") from exc
    except VioBadRequestError as exc:
        raise HTTPException(status_code=400, detail=f"VIO Requisição inválida: {exc}") from exc
    except VioNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"VIO Não encontrado: {exc}") from exc
    except VioEnvironmentError as exc:
        raise HTTPException(status_code=400, detail=f"VIO Ambiente incompatível: {exc}") from exc
    except VioApiError as exc:
        raise HTTPException(status_code=502, detail=f"Erro na API VIO: {exc}") from exc


@router.post("/api/consultar-imagem")
async def consultar_imagem_upload(
    file: Annotated[UploadFile, File(description="Arquivo de imagem contendo o QR Code (PNG, JPG, etc)")],
    wait: Annotated[bool, Form()] = True,
    incluir_qrcode_base64: Annotated[bool, Form()] = False,
):
    """Extrai bytes binários do QR, reconstitui e consulta na API VIO (Serpro)."""
    conteudo_arquivo = await file.read()
    if not conteudo_arquivo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    dados_binarios = extrair_bytes_qrcode(conteudo_arquivo)
    b64_qr = gerar_qrcode_base64(dados_binarios)
    resultado_vio = consultar_vio_com_base64(b64_qr, wait=wait)
    placa = extrair_placa_de_dicionario(resultado_vio)

    payload = {
        "dados_vio": resultado_vio,
        "bytes_extraidos_total": len(dados_binarios),
    }
    if incluir_qrcode_base64:
        payload["qrcode_reconstituido_base64"] = b64_qr

    return {
        "sucesso": True,
        "validador": "integrador-vio",
        "arquivo": file.filename,
        "placa": placa,
        "payload": payload,
    }


@router.post("/api/consultar")
def consultar_json(req: ConsultaVioRequest):
    """Recebe JSON, reconstitui QR Code e consulta na API VIO (Serpro)."""
    dados_binarios: bytes | None = None

    if req.imagem_base64:
        b64 = req.imagem_base64
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw_img = base64.b64decode(b64)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Base64 inválido: {exc}") from exc
        dados_binarios = extrair_bytes_qrcode(raw_img)

    elif req.caminho_imagem:
        path = Path(req.caminho_imagem)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {path}")
        dados_binarios = extrair_bytes_qrcode(path.read_bytes())

    elif req.texto:
        b64_qr = gerar_qrcode_base64(req.texto)
        resultado_vio = consultar_vio_com_base64(b64_qr, wait=req.wait)
        placa = extrair_placa_de_dicionario(resultado_vio)
        payload = {
            "origem": "texto",
            "dados_vio": resultado_vio,
        }
        if req.incluir_qrcode_base64:
            payload["qrcode_reconstituido_base64"] = b64_qr
        return {
            "sucesso": True,
            "validador": "integrador-vio",
            "arquivo": None,
            "placa": placa,
            "payload": payload,
        }

    else:
        raise HTTPException(
            status_code=400,
            detail="Informe ao menos um: imagem_base64, caminho_imagem ou texto.",
        )

    b64_qr = gerar_qrcode_base64(dados_binarios)
    resultado_vio = consultar_vio_com_base64(b64_qr, wait=req.wait)
    placa = extrair_placa_de_dicionario(resultado_vio)
    nome_arquivo = Path(req.caminho_imagem).name if req.caminho_imagem else "base64_upload.png"

    payload = {
        "dados_vio": resultado_vio,
        "bytes_extraidos_total": len(dados_binarios),
    }
    if req.incluir_qrcode_base64:
        payload["qrcode_reconstituido_base64"] = b64_qr

    return {
        "sucesso": True,
        "validador": "integrador-vio",
        "arquivo": nome_arquivo,
        "placa": placa,
        "payload": payload,
    }
