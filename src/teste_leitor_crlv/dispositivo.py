from __future__ import annotations

import logging
from typing import Any

import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)


def detectar_acelerador_torch() -> tuple[bool, str]:
    """Detecta se há GPU compatível com PyTorch (CUDA ou MPS). Retorna (tem_gpu, nome_dispositivo)."""
    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cuda"
        logger.info("Aceleração GPU detectada para PyTorch/EasyOCR: %s", nome)
        return True, nome

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Aceleração Apple Silicon (MPS) detectada para PyTorch.")
        return True, "mps"

    logger.debug("Nenhuma GPU dedicada detectada para PyTorch. Utilizando CPU.")
    return False, "cpu"


def detectar_providers_onnx() -> list[str]:
    """Retorna lista ordenada de Execution Providers disponíveis no ONNX Runtime.

    Prioriza: CUDA (NVIDIA) > DirectML (AMD/Intel no Windows) > CPU (Universal).
    """
    disponiveis = set(ort.get_available_providers())
    prioridades = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    providers = [p for p in prioridades if p in disponiveis]
    if not providers:
        providers = ["CPUExecutionProvider"]

    logger.info("Execution Providers ONNX selecionados: %s", providers)
    return providers


def obter_info_aceleradores() -> dict[str, Any]:
    """Retorna metadados dos aceleradores de hardware disponíveis no ambiente."""
    tem_gpu_torch, nome_gpu_torch = detectar_acelerador_torch()
    providers_onnx = detectar_providers_onnx()

    return {
        "pytorch": {
            "gpu_disponivel": tem_gpu_torch,
            "dispositivo": nome_gpu_torch,
        },
        "onnxruntime": {
            "provider_ativo": providers_onnx[0],
            "providers_disponiveis": providers_onnx,
        },
    }
