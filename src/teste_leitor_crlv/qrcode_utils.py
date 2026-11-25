import base64
import io
from typing import Any

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


def extrair_placa_de_dicionario(dados: Any) -> str | None:
    """Varre recursivamente campos de retorno para extrair o valor da placa de veículo se presente."""
    if not isinstance(dados, dict):
        return None

    # 1. Procura direta nas chaves do primeiro nível
    for k, v in dados.items():
        if (
            isinstance(k, str)
            and k.lower() in ("placa", "placa_veiculo")
            and isinstance(v, str)
            and len(v.strip()) == 7
        ):
            return v.strip().upper()

    # 2. Varre dicionários internos conhecidos
    for subchave in ("fields", "raw_fields", "dados", "analise", "document", "veiculo", "resultado"):
        sub = dados.get(subchave)
        if isinstance(sub, dict):
            res = extrair_placa_de_dicionario(sub)
            if res:
                return res

    # 3. Busca recursiva genérica
    for v in dados.values():
        if isinstance(v, dict):
            res = extrair_placa_de_dicionario(v)
            if res:
                return res

    return None
