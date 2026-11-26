import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .dispositivo import detectar_providers_onnx

DIR_PROJETO = Path(__file__).resolve().parent.parent.parent
DIR_MODELO_PADRAO = DIR_PROJETO / "treinamento" / "modelos"


class ReconhecedorCRNNLocal:
    """Executor de inferência para o modelo CRNN treinado localmente via ONNXRuntime."""

    def __init__(self, modelo_path: str | None = None, vocab_path: str | None = None):
        self.modelo_path = Path(modelo_path or os.getenv("CRNN_MODEL_PATH") or (DIR_MODELO_PADRAO / "crnn_crlv.onnx"))
        self.vocab_path = Path(vocab_path or os.getenv("CRNN_VOCAB_PATH") or (DIR_MODELO_PADRAO / "vocabulario.json"))

        if not self.modelo_path.exists() or not self.vocab_path.exists():
            raise FileNotFoundError(f"Modelo local não encontrado em: {self.modelo_path}")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}

        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # Silencia warnings do ONNXRuntime
        providers = detectar_providers_onnx()
        self.session = ort.InferenceSession(str(self.modelo_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def modelo_disponivel(cls, caminho_custom: str | None = None) -> bool:
        """Verifica se os arquivos do modelo treinado localmente estão presentes."""
        base = Path(caminho_custom or os.getenv("CRNN_MODEL_PATH") or (DIR_MODELO_PADRAO / "crnn_crlv.onnx"))
        vocab = Path(os.getenv("CRNN_VOCAB_PATH") or (DIR_MODELO_PADRAO / "vocabulario.json"))
        return base.exists() and vocab.exists()

    def reconhecer_crop(self, crop: Image.Image) -> tuple[str, float]:
        """Recebe PIL.Image do recorte de texto e retorna (texto_decodificado, score_estimado)."""
        img_l = crop.convert("L")
        w, h = img_l.size
        nova_w = max(16, min(256, int(w * (32.0 / max(h, 1)))))
        img_l = img_l.resize((nova_w, 32), Image.Resampling.BILINEAR)

        tensor = np.zeros((1, 1, 32, 256), dtype=np.float32)
        arr = (np.array(img_l, dtype=np.float32) / 255.0 - 0.5) / 0.5
        tensor[0, 0, :, :nova_w] = arr

        outputs = self.session.run(None, {self.input_name: tensor})
        # outputs[0] shape: (T, 1, num_classes)
        log_probs = outputs[0]
        pred_indices = log_probs.squeeze(1).argmax(axis=-1)

        chars = []
        prev = None
        for idx in pred_indices:
            if idx != 0 and idx != prev:
                chars.append(self.idx_to_char.get(idx, ""))
            prev = idx

        texto_decodificado = "".join(chars)
        return texto_decodificado, 0.95


_reconhecedor_singleton: ReconhecedorCRNNLocal | None = None


def obter_reconhecedor_local() -> ReconhecedorCRNNLocal | None:
    """Retorna instância singleton do reconhecedor treinado localmente se existir."""
    global _reconhecedor_singleton
    if (_reconhecedor_singleton is None) and (ReconhecedorCRNNLocal.modelo_disponivel()):
        _reconhecedor_singleton = ReconhecedorCRNNLocal()
    return _reconhecedor_singleton
