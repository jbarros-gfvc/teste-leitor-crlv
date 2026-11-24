import base64
import binascii
from pathlib import Path
from typing import Annotated

from digital_document_checker import DigitalDocumentChecker
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .qrcode_utils import extrair_bytes_qrcode

router = APIRouter(tags=["Local / Offline (Digital Document Checker)"])


class ValidacaoLocalRequest(BaseModel):
    imagem_base64: str | None = Field(
        default=None,
        description="Imagem do documento em Base64 ou Data URI contendo o QR Code",
    )
    caminho_imagem: str | None = Field(
        default=None,
        description="Caminho do arquivo de imagem local contendo o QR Code",
    )
    qrcode_base64: str | None = Field(
        default=None,
        description="Base64 direto dos bytes brutos contidos no QR Code",
    )
    include_raw: bool = Field(
        default=False,
        description="Inclui dados brutos (signed_data, foto base64) na resposta",
    )


def validar_local_com_checker(b64_qr: str, include_raw: bool = False) -> dict:
    """Valida assinatura e extrai dados localmente via digital-document-checker."""
    checker = DigitalDocumentChecker()
    res = checker.parse_base64(b64_qr)
    return res.to_dict(include_raw=include_raw)


@router.post("/api/validar-localmente-imagem")
async def validar_localmente_upload(
    file: Annotated[UploadFile, File(description="Arquivo de imagem (CNH, CIN, CRLV, etc)")],
    include_raw: Annotated[bool, Form()] = False,
):
    """Extrai bytes do QR Code da imagem enviada, extrai Base64 e valida localmente via digital-document-checker."""
    conteudo_arquivo = await file.read()
    if not conteudo_arquivo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    dados_binarios = extrair_bytes_qrcode(conteudo_arquivo)
    b64_qr = base64.b64encode(dados_binarios).decode("ascii")
    resultado_dict = validar_local_com_checker(b64_qr, include_raw=include_raw)

    return {
        "sucesso": resultado_dict["status"]["is_parsed"],
        "validador": "digital-document-checker (local/offline)",
        "arquivo": file.filename,
        "bytes_extraidos_total": len(dados_binarios),
        "qrcode_extraido_base64": b64_qr,
        "resultado": resultado_dict,
    }


@router.post("/api/validar-localmente")
def validar_localmente_json(req: ValidacaoLocalRequest):
    """Recebe imagem ou QR Code em Base64 e valida autenticidade/assinatura localmente."""
    b64_qr: str | None = None
    bytes_extraidos_total: int = 0

    if req.qrcode_base64:
        b64_qr = req.qrcode_base64
        try:
            bytes_extraidos_total = len(base64.b64decode(b64_qr))
        except (binascii.Error, ValueError):
            bytes_extraidos_total = len(b64_qr)

    elif req.imagem_base64:
        b64 = req.imagem_base64
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw_img = base64.b64decode(b64)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Base64 da imagem inválido: {exc}") from exc
        dados_binarios = extrair_bytes_qrcode(raw_img)
        bytes_extraidos_total = len(dados_binarios)
        b64_qr = base64.b64encode(dados_binarios).decode("ascii")

    elif req.caminho_imagem:
        path = Path(req.caminho_imagem)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {path}")
        dados_binarios = extrair_bytes_qrcode(path.read_bytes())
        bytes_extraidos_total = len(dados_binarios)
        b64_qr = base64.b64encode(dados_binarios).decode("ascii")

    else:
        raise HTTPException(
            status_code=400,
            detail="Informe ao menos um: imagem_base64, caminho_imagem ou qrcode_base64.",
        )

    resultado_dict = validar_local_com_checker(b64_qr, include_raw=req.include_raw)

    return {
        "sucesso": resultado_dict["status"]["is_parsed"],
        "validador": "digital-document-checker (local/offline)",
        "bytes_extraidos_total": bytes_extraidos_total,
        "qrcode_extraido_base64": b64_qr,
        "resultado": resultado_dict,
    }
