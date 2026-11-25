import logging
import threading
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI

from .analise_local import (
    router as local_router,
)
from .analise_ocr import (
    router as ocr_router,
)
from .analise_paddleocr import (
    obter_reader_paddle,
)
from .analise_paddleocr import (
    router as paddle_router,
)
from .analise_paralela import (
    router as paralelo_router,
)
from .analise_vio import (
    router as vio_router,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pré-aquece os modelos de inferência ONNX em background thread para eliminar cold-start."""

    def _warmup():
        try:
            reader = obter_reader_paddle()
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            reader.predict(dummy_img)
            logger.info("Warm-up concluído com sucesso.")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Warm-up silencioso: %s", exc)

    threading.Thread(target=_warmup, daemon=True, name="model_warmup").start()
    yield


app = FastAPI(
    title="API Leitor CRLV (Digital Document Checker, EasyOCR, PaddleOCR & VIO)",
    description="Validação via API Serpro VIO, Digital Document Checker (offline), EasyOCR, PaddleOCR e corrida multithread de alta velocidade",
    version="0.8.0",
    lifespan=lifespan,
)

# Registra os módulos de análise por responsabilidade
app.include_router(vio_router)
app.include_router(local_router)
app.include_router(ocr_router)
app.include_router(paddle_router)
app.include_router(paralelo_router)


@app.get("/", tags=["Status"])
def raiz():
    return {
        "status": "online",
        "modos": {
            "analise_rapida_paralela": {
                "upload": "POST /api/analisar-rapido-imagem (multipart/form-data)",
                "json": "POST /api/analisar-rapido (JSON)",
            },
            "remoto_api_vio": {
                "upload": "POST /api/consultar-imagem (multipart/form-data)",
                "json": "POST /api/consultar (JSON)",
            },
            "local_offline_checker": {
                "upload": "POST /api/validar-localmente-imagem (multipart/form-data)",
                "json": "POST /api/validar-localmente (JSON)",
            },
            "leitura_documentos_easyocr": {
                "upload": "POST /api/ocr-crlv-imagem (multipart/form-data)",
                "json": "POST /api/ocr-crlv (JSON)",
            },
            "leitura_documentos_paddleocr": {
                "upload": "POST /api/paddleocr-crlv-imagem (multipart/form-data)",
                "json": "POST /api/paddleocr-crlv (JSON)",
            },
        },
        "docs": "/docs",
    }


def main():
    uvicorn.run("teste_leitor_crlv.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
