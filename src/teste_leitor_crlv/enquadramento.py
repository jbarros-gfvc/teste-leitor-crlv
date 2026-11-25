from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import zxingcpp
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


@dataclass
class MetadadosEnquadramento:
    metodo: str  # "qrcode", "contorno_papel", "original"
    rotacao_graus: int
    crop_aplicado: bool
    dimensoes_originais: tuple[int, int]  # (largura, altura)
    dimensoes_finais: tuple[int, int]  # (largura, altura)
    bbox_crop: tuple[int, int, int, int] | None  # (x_min, y_min, x_max, y_max)
    qrcode_info: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metodo": self.metodo,
            "rotacao_graus": self.rotacao_graus,
            "crop_aplicado": self.crop_aplicado,
            "dimensoes_originais": {
                "largura": self.dimensoes_originais[0],
                "altura": self.dimensoes_originais[1],
            },
            "dimensoes_finais": {
                "largura": self.dimensoes_finais[0],
                "altura": self.dimensoes_finais[1],
            },
            "bbox_crop": (
                {
                    "x_min": self.bbox_crop[0],
                    "y_min": self.bbox_crop[1],
                    "x_max": self.bbox_crop[2],
                    "y_max": self.bbox_crop[3],
                }
                if self.bbox_crop
                else None
            ),
            "qrcode_detectado": self.qrcode_info is not None,
        }


def _corrigir_exif(imagem: np.ndarray | Image.Image) -> np.ndarray:
    """Corrige orientação EXIF comum em fotos tiradas de celular."""
    if isinstance(imagem, np.ndarray):
        pil_img = Image.fromarray(imagem)
    else:
        pil_img = imagem
    pil_corrigida = ImageOps.exif_transpose(pil_img)
    return np.array(pil_corrigida)


def _detectar_e_orientar_por_qrcode(
    img_bgr: np.ndarray,
) -> tuple[np.ndarray, int, zxingcpp.Barcode | None]:
    """Detecta QR Code via zxing-cpp e alinha a imagem para 0 graus horizontal."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    barcode = zxingcpp.read_barcode(pil_img)
    if not barcode or not barcode.valid:
        return img_bgr, 0, None

    rot = barcode.orientation
    rot_aplicada = 0
    img_orientada = img_bgr

    # Rotaciona para deixar o documento alinhado horizontalmente
    if 45 <= rot <= 135:
        img_orientada = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot_aplicada = 90
    elif 135 < rot <= 225:
        img_orientada = cv2.rotate(img_bgr, cv2.ROTATE_180)
        rot_aplicada = 180
    elif 225 < rot <= 315:
        img_orientada = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        rot_aplicada = 270

    # Se houve rotação, re-lê o QR Code na imagem já orientada
    if rot_aplicada != 0:
        pil_rot = Image.fromarray(cv2.cvtColor(img_orientada, cv2.COLOR_BGR2RGB))
        barcode_orientado = zxingcpp.read_barcode(pil_rot)
        if barcode_orientado and barcode_orientado.valid:
            barcode = barcode_orientado

    return img_orientada, rot_aplicada, barcode


def _calcular_roi_crlv_por_qrcode(img_shape: tuple[int, ...], barcode: zxingcpp.Barcode) -> tuple[int, int, int, int]:
    """Calcula a Região de Interesse (ROI) do CRLV baseando-se na posição do QR Code à direita.

    No layout do CRLV-e (cartão):
    - Topo: faixa escura/verde ('REPÚBLICA FEDERATIVA DO BRASIL') e brasão à esquerda
    - Direita: QR Code + faixa vertical 'Valide este QRCode com app Vio'
    - Esquerda: dados do veículo (RENAVAM, Placa, Marca, etc.) à esquerda do QR Code
    """
    h_img, w_img = img_shape[:2]
    pts = [
        barcode.position.top_left,
        barcode.position.top_right,
        barcode.position.bottom_right,
        barcode.position.bottom_left,
    ]
    min_x = min(p.x for p in pts)
    max_x = max(p.x for p in pts)
    min_y = min(p.y for p in pts)
    max_y = max(p.y for p in pts)

    qr_w = max(max_x - min_x, 10)
    qr_h = max(max_y - min_y, 10)

    # Margens proporcionais ao tamanho do QR Code
    # À esquerda: dados do veículo ocupam cerca de 1.8x a 2.0x a largura do QR
    x_min = max(0, int(min_x - 1.95 * qr_w))
    # À direita: QR Code + margem do texto 'Valide este QRCode com app Vio' (~0.35x qr_w)
    x_max = min(w_img, int(max_x + 0.35 * qr_w))
    # Acima: faixa superior escura 'REPÚBLICA FEDERATIVA DO BRASIL' (~0.95x qr_h)
    y_min = max(0, int(min_y - 0.95 * qr_h))
    # Abaixo: numeração do CRV e caixas de assinatura (~1.20x qr_h)
    y_max = min(h_img, int(max_y + 1.25 * qr_h))

    return x_min, y_min, x_max, y_max


def _detectar_contorno_papel(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Busca contorno retangular dominante (papel/cartão) quando não há QR Code legível."""
    h, w = img_bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    small_w = max(int(w * scale), 10)
    small_h = max(int(h * scale), 10)
    small = cv2.resize(img_bgr, (small_w, small_h))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor_box = None
    max_area = 0
    area_minima = (small_w * small_h) * 0.18  # pelo menos 18% da imagem

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_minima and area > max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                max_area = area
                melhor_box = approx

    if melhor_box is not None:
        pts = melhor_box.reshape(4, 2) / scale
        x0 = max(0, int(min(p[0] for p in pts)))
        x1 = min(w, int(max(p[0] for p in pts)))
        y0 = max(0, int(min(p[1] for p in pts)))
        y1 = min(h, int(max(p[1] for p in pts)))
        if (x1 - x0) > 100 and (y1 - y0) > 100:
            return x0, y0, x1, y1

    return None


def otimizar_enquadramento_crlv(
    imagem_rgb: np.ndarray,
    margem_seguranca: float = 0.02,
) -> tuple[np.ndarray, MetadadosEnquadramento]:
    """Executa detecção de âncoras (QR Code Vio, faixa superior e contorno) e recorta a área útil do CRLV.

    Retorna a imagem recortada/alinhada em RGB e o objeto de metadados.
    """
    # 1. Garante correção EXIF
    img_corrigida = _corrigir_exif(imagem_rgb)
    h_orig, w_orig = img_corrigida.shape[:2]
    img_bgr = cv2.cvtColor(img_corrigida, cv2.COLOR_RGB2BGR)

    # 2. Prioridade 1: Âncora QR Code (fornece rotação exata e limite direito)
    img_orientada, rot_aplicada, barcode = _detectar_e_orientar_por_qrcode(img_bgr)
    h_orientada, w_orientada = img_orientada.shape[:2]

    if barcode is not None and barcode.valid:
        x_min, y_min, x_max, y_max = _calcular_roi_crlv_por_qrcode(img_orientada.shape, barcode)

        # Aplica margem extra de segurança
        folga_w = int((x_max - x_min) * margem_seguranca)
        folga_h = int((y_max - y_min) * margem_seguranca)
        x_min = max(0, x_min - folga_w)
        y_min = max(0, y_min - folga_h)
        x_max = min(w_orientada, x_max + folga_w)
        y_max = min(h_orientada, y_max + folga_h)

        crop_bgr = img_orientada[y_min:y_max, x_min:x_max]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h_final, w_final = crop_rgb.shape[:2]

        qrcode_info = {
            "bytes_length": len(barcode.bytes) if barcode.bytes else 0,
            "formato": str(barcode.format),
        }

        meta = MetadadosEnquadramento(
            metodo="qrcode_vio",
            rotacao_graus=rot_aplicada,
            crop_aplicado=True,
            dimensoes_originais=(w_orig, h_orig),
            dimensoes_finais=(w_final, h_final),
            bbox_crop=(x_min, y_min, x_max, y_max),
            qrcode_info=qrcode_info,
        )
        return crop_rgb, meta

    # 3. Prioridade 2: Contorno do documento quando QR Code não está legível/rasurado
    bbox_papel = _detectar_contorno_papel(img_orientada)
    if bbox_papel is not None:
        px0, py0, px1, py1 = bbox_papel
        crop_bgr = img_orientada[py0:py1, px0:px1]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h_final, w_final = crop_rgb.shape[:2]

        meta = MetadadosEnquadramento(
            metodo="contorno_documento",
            rotacao_graus=rot_aplicada,
            crop_aplicado=True,
            dimensoes_originais=(w_orig, h_orig),
            dimensoes_finais=(w_final, h_final),
            bbox_crop=(px0, py0, px1, py1),
            qrcode_info=None,
        )
        return crop_rgb, meta

    # 4. Fallback: Mantém imagem orientada integral
    img_rgb_final = cv2.cvtColor(img_orientada, cv2.COLOR_BGR2RGB)
    meta = MetadadosEnquadramento(
        metodo="original_sem_crop",
        rotacao_graus=rot_aplicada,
        crop_aplicado=False,
        dimensoes_originais=(w_orig, h_orig),
        dimensoes_finais=(w_orientada, h_orientada),
        bbox_crop=None,
        qrcode_info=None,
    )
    return img_rgb_final, meta
