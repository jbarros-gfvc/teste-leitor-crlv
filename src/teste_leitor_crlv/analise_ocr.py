import base64
import binascii
import io
import re
from pathlib import Path
from typing import Annotated

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageOps

try:
    import pymupdf as fitz
except ImportError:
    import fitz
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .enquadramento import otimizar_enquadramento_crlv

router = APIRouter(tags=["Leitura OCR de Documentos (CRLV)"])

_ocr_reader: easyocr.Reader | None = None


def obter_reader_ocr() -> easyocr.Reader:
    """Retorna instância singleton do EasyOCR Reader (pt, cpu)."""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["pt"], gpu=False)
    return _ocr_reader


class OcrCrlvRequest(BaseModel):
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


def carregar_imagem(origem: str | Path | bytes, nome_arquivo: str = "") -> np.ndarray:
    """Carrega imagem ou primeira página de PDF para array numpy RGB com correção EXIF."""
    try:
        if isinstance(origem, (str, Path)):
            caminho_str = str(origem).lower()
            if caminho_str.endswith(".pdf"):
                doc = fitz.open(str(origem))
                pix = doc[0].get_pixmap(dpi=250)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                return np.array(img)
            # ImageOps.exif_transpose corrige fotos de celular salvas com rotação EXIF
            img = ImageOps.exif_transpose(Image.open(origem)).convert("RGB")
            return np.array(img)
        elif isinstance(origem, bytes):
            if origem.startswith(b"%PDF") or (nome_arquivo and nome_arquivo.lower().endswith(".pdf")):
                doc = fitz.open(stream=origem, filetype="pdf")
                pix = doc[0].get_pixmap(dpi=250)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                return np.array(img)
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(origem))).convert("RGB")
            return np.array(img)
        else:
            raise TypeError("Origem inválida para carregar imagem.")
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem/PDF: {exc}") from exc


def corrigir_orientacao(imagem: np.ndarray, reader: easyocr.Reader | None = None) -> np.ndarray:
    """Testa rotação caso o documento esteja deitado (90° ou 270°)."""
    if reader is None:
        reader = obter_reader_ocr()

    def _extrair_texto_plano(res):
        linhas = []
        for item in res:
            if isinstance(item, (tuple, list)):
                if len(item) >= 2 and isinstance(item[1], str):
                    linhas.append(item[1])
                elif len(item) > 0 and isinstance(item[0], str):
                    linhas.append(item[0])
            elif isinstance(item, str):
                linhas.append(item)
        return " ".join(linhas).upper()

    textos_0 = _extrair_texto_plano(reader.readtext(imagem, detail=0))
    if "PLACA" in textos_0 or "RENAVAM" in textos_0:
        return imagem

    for rotacao in [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE]:
        img_rot = cv2.rotate(imagem, rotacao)
        textos_rot = _extrair_texto_plano(reader.readtext(img_rot, detail=0))
        if "PLACA" in textos_rot or "RENAVAM" in textos_rot:
            return img_rot

    return imagem


def delimitar_documento_pelos_textos(resultados_ocr):
    """
    Encontra o retângulo envolvente (bounding box) de todo o conteúdo textual do documento.
    Ignora fundos de mesa, cadernos ou bordas brancas extras.
    """
    todos_x = []
    todos_y = []

    for bbox, texto, conf in resultados_ocr:
        if conf > 0.25:  # Ignora ruídos muito fracos do fundo
            for ponto in bbox:
                todos_x.append(ponto[0])
                todos_y.append(ponto[1])

    if not todos_x or not todos_y:
        return 0, 0, 1, 1

    return min(todos_x), min(todos_y), max(todos_x), max(todos_y)


def dentro_da_area(x_norm: float, y_norm: float, limites: tuple, margem: float = 0.10) -> bool:
    """Verifica se o ponto central (x, y) está dentro do retângulo esperado + margem de erro."""
    x_min, y_min, x_max, y_max = limites
    return (x_min - margem) <= x_norm <= (x_max + margem) and (y_min - margem) <= y_norm <= (y_max + margem)


def corrigir_placa(placa_bruta: str) -> str:
    """Corrige trocas comuns de OCR respeitando o padrão LLLNLNN ou LLLNNNN."""
    if len(placa_bruta) != 7:
        return placa_bruta

    mapa_para_letra = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
    mapa_para_numero = {"O": "0", "Q": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}

    chars = list(placa_bruta)
    # 3 primeiros são sempre letras
    for i in [0, 1, 2]:
        chars[i] = mapa_para_letra.get(chars[i], chars[i])
    # 4º, 6º e 7º são sempre números
    for i in [3, 5, 6]:
        chars[i] = mapa_para_numero.get(chars[i], chars[i])

    return "".join(chars)


def analisar_crlv_com_posicao(
    origem: str | Path | bytes | np.ndarray,
    margem_erro: float = 0.10,
    nome_arquivo: str = "",
    reader: easyocr.Reader | None = None,
    auto_orientar: bool = True,
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
        # Se o enquadramento orientou a imagem pelo QR Code, dispensa busca exaustiva
        if meta.rotacao_graus != 0 or meta.metodo == "qrcode_vio":
            auto_orientar = False

    if reader is None:
        reader = obter_reader_ocr()

    if auto_orientar:
        imagem = corrigir_orientacao(imagem, reader=reader)

    # Executa o OCR na imagem orientada
    resultados = reader.readtext(imagem, detail=1)

    # 1. Delimita a área real do documento com base nos textos encontrados
    doc_x_min, doc_y_min, doc_x_max, doc_y_max = delimitar_documento_pelos_textos(resultados)
    doc_largura = max(doc_x_max - doc_x_min, 1)
    doc_altura = max(doc_y_max - doc_y_min, 1)

    # 2. Zonas relativas AO DOCUMENTO DELIMITADO (0.0 a 1.0)
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
        # Normaliza texto (remove acentos básicos)
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

        # Centro do texto relativo APENAS à área delimitada do documento
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

    # Considera válido se pelo menos 4 das 6 âncoras estiverem na posição correta
    layout_valido = total_validadas >= 4

    # 3. Extração da Placa (aceita caracteres confundíveis e corrige depois)
    padrao_placa = r"\b([A-Z0-9]{3})[-\s]?([0-9A-Z]{4})\b"
    placa_encontrada = None

    for i, linha in enumerate(linhas_texto):
        if "PLACA" in linha:
            # Busca nos blocos logo após "PLACA"
            trecho_busca = " ".join(linhas_texto[i : i + 4])
            trecho_busca = trecho_busca.replace("PLACA", "").replace("EXERCICIO", "")
            match = re.search(padrao_placa, trecho_busca)
            if match:
                candidata = corrigir_placa(f"{match.group(1)}{match.group(2)}")
                if re.match(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$", candidata):
                    placa_encontrada = candidata
                    break

    # Fallback global caso o rótulo PLACA tenha falhado
    if not placa_encontrada:
        for candidato in re.findall(r"\b[A-Z0-9]{7}\b", texto_completo):
            corrigida = corrigir_placa(candidato)
            if re.match(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$", corrigida):
                placa_encontrada = corrigida
                break

    # 4. RENAVAM (11 dígitos)
    renavam_match = re.search(r"\b\d{11}\b", texto_completo)

    return {
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


# Alias de compatibilidade com o snippet fornecido
analisar_crlv_cartao = analisar_crlv_com_posicao


@router.post("/api/ocr-crlv-imagem")
async def ocr_crlv_upload(
    file: Annotated[UploadFile, File(description="Arquivo de documento CRLV (PDF, PNG, JPG, etc)")],
    margem_erro: Annotated[float, Form()] = 0.10,
):
    """Lê documento CRLV via OCR (EasyOCR/PyMuPDF) e valida layout por posição de âncoras."""
    conteudo_arquivo = await file.read()
    if not conteudo_arquivo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    resultado = analisar_crlv_com_posicao(
        origem=conteudo_arquivo,
        margem_erro=margem_erro,
        nome_arquivo=file.filename or "",
    )
    return {
        "sucesso": bool(resultado.get("formato_crlv_valido", False)),
        "validador": "easyocr",
        "arquivo": file.filename,
        "placa": resultado.get("placa"),
        "payload": resultado,
    }


@router.post("/api/ocr-crlv")
def ocr_crlv_json(req: OcrCrlvRequest):
    """Recebe imagem ou PDF em Base64 ou caminho de arquivo e analisa layout CRLV via OCR."""
    if req.caminho_arquivo:
        path = Path(req.caminho_arquivo)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {path}")
        resultado = analisar_crlv_com_posicao(
            origem=str(path),
            margem_erro=req.margem_erro,
        )
    elif req.imagem_base64 or req.arquivo_base64:
        b64 = req.imagem_base64 or req.arquivo_base64
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(b64)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Base64 inválido: {exc}") from exc

        resultado = analisar_crlv_com_posicao(
            origem=raw_bytes,
            margem_erro=req.margem_erro,
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
        "validador": "easyocr",
        "arquivo": nome_arquivo,
        "placa": resultado.get("placa"),
        "payload": resultado,
    }
