import base64
import io

import segno
import zxingcpp
from fastapi import HTTPException
from PIL import Image


def extrair_bytes_qrcode(imagem_bytes: bytes) -> bytes:
    """Abre a imagem e extrai os bytes binários brutos do QR Code via zxing-cpp."""
    try:
        pil_img = Image.open(io.BytesIO(imagem_bytes))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Arquivo de imagem inválido: {exc}") from exc

    resultado = zxingcpp.read_barcode(pil_img)
    if not resultado or not resultado.valid:
        raise HTTPException(
            status_code=422,
            detail="Nenhum QR Code válido foi detectado na imagem fornecida.",
        )
    return resultado.bytes


def gerar_qrcode_base64(conteudo: bytes | str) -> str:
    """Gera QR code em PNG a partir de bytes ou texto e retorna Base64."""
    buffer = io.BytesIO()
    qr = segno.make(conteudo)
    qr.save(buffer, kind="png", scale=5)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
