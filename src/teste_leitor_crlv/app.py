import uvicorn
from fastapi import FastAPI

from .analise_local import (
    router as local_router,
)
from .analise_ocr import (
    router as ocr_router,
)
from .analise_paddleocr import (
    router as paddle_router,
)
from .analise_vio import (
    router as vio_router,
)

app = FastAPI(
    title="API Integrador VIO & Digital Document Checker & OCR (EasyOCR & PaddleOCR)",
    description="Validação via API Serpro VIO (remota), Digital Document Checker (offline) e OCR de documentos (EasyOCR e PaddleOCR)",
    version="0.6.0",
)

# Registra os módulos de análise por responsabilidade
app.include_router(vio_router)
app.include_router(local_router)
app.include_router(ocr_router)
app.include_router(paddle_router)


@app.get("/", tags=["Status"])
def raiz():
    return {
        "status": "online",
        "modos": {
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
