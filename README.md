# Teste Leitor CRLV (VIO, Digital Document Checker & OCR)

Documentação técnica completa e detalhada por módulo em: [DOCUMENTACAO_ANALISADORES.md](file:///C:/Users/user/Documents/teste-integrador-vio-api/DOCUMENTACAO_ANALISADORES.md)

API FastAPI com suporte a quatro métodos de validação e leitura de documentos:

1. **Remoto (API Serpro VIO)** via [`integrador-vio`](https://github.com/kejrra/integrador-vio): Consulta oficial com chave de API.
2. **Local / Offline (Contra Ponto)** via [`digital-document-checker`](https://github.com/helviojunior/digital-document-checker): Valida assinaturas RSA/ECDSA/JWS localmente usando chaves públicas dos emissores (SENATRAN/SERPRO/MJSP), sem custos de API ou rede.
3. **Leitura de Documentos via EasyOCR**: Análise geométrica de âncoras (República Federativa, Certificado, RENAVAM, Placa, Exercício, CRV), correção de rotação/EXIF, extração de Placa e validação de RENAVAM.
4. **Leitura de Documentos via PaddleOCR (ONNX/PP-OCRv6)**: Reconhecimento rápido e preciso com suporte nativo a treinamento e fine-tuning.

---

## 1. Execução

```bash
uv run teste-leitor-crlv
# ou
uv run uvicorn teste_leitor_crlv.app:app --reload --port 8000
```

Swagger UI interativo: `http://localhost:8000/docs`

---

## 2. Endpoints Locais / Offline (`digital-document-checker`)

- **POST** `/api/validar-localmente-imagem` (Multipart Form)
- **POST** `/api/validar-localmente` (JSON)

---

## 3. Endpoints Remotos (`integrador-vio`)

- **POST** `/api/consultar-imagem` (Multipart Form)
- **POST** `/api/consultar` (JSON)

---

## 4. Endpoints Leitura EasyOCR

- **POST** `/api/ocr-crlv-imagem` (Multipart Form)
- **POST** `/api/ocr-crlv` (JSON)

---

## 5. Endpoints Leitura PaddleOCR (Novo)

- **POST** `/api/paddleocr-crlv-imagem` (Multipart Form)
- **POST** `/api/paddleocr-crlv` (JSON)
