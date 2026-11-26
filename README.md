# Teste Leitor CRLV (VIO, Digital Document Checker & OCR)

Documentação técnica completa e detalhada por módulo em: [DOCUMENTACAO_ANALISADORES.md](file:///C:/Users/user/Documents/teste-integrador-vio-api/DOCUMENTACAO_ANALISADORES.md)

API FastAPI com suporte a quatro métodos de validação e leitura de documentos:

1. **Remoto (API Serpro VIO)** via [`integrador-vio`](https://github.com/kejrra/integrador-vio): Consulta oficial com chave de API.
2. **Local / Offline (Contra Ponto)** via [`digital-document-checker`](https://github.com/helviojunior/digital-document-checker): Valida assinaturas RSA/ECDSA/JWS localmente usando chaves públicas dos emissores (SENATRAN/SERPRO/MJSP), sem custos de API ou rede.
3. **Leitura de Documentos via EasyOCR**: Análise geométrica de âncoras (República Federativa, Certificado, RENAVAM, Placa, Exercício, CRV), correção de rotação/EXIF, extração de Placa e validação de RENAVAM.
4. **Leitura de Documentos via PaddleOCR (ONNX/PP-OCRv6)**: Reconhecimento rápido e preciso com suporte nativo a treinamento e fine-tuning.
5. **Enquadramento Inteligente com Âncoras**: Recorte automático da área útil do documento guiado pela posição do QR Code à direita (e texto Vio) e faixa escura superior (brasão), eliminando mesas/fundos e corrigindo rotações de 90°/180°/270° instantaneamente.
6. **Auto-Detecção de Aceleração por GPU**: Ativação transparente de GPU (CUDA / DirectML) se disponível no ambiente com fallback automático para CPU.

---

## 1. Execução

```bash
uv run teste-leitor-crlv
# ou
uv run uvicorn teste_leitor_crlv.app:app --reload --port 8000
```

Swagger UI interativo: `http://localhost:8000/docs`

### Contrato Padronizado de Resposta (JSON)
Todos os endpoints retornam a mesma estrutura raiz:
```json
{
  "sucesso": true,
  "validador": "digital-document-checker",
  "arquivo": "crlv.jpg",
  "placa": "ABC1D23",
  "payload": { ... }
}
```

---

## 2. Endpoints Análise Rápida Paralela (Fast-Path & Multithread)

Executa Fast-Path via DigitalDocumentChecker (<50ms). Se o QR Code for ausente ou rasurado, dispara PaddleOCR e EasyOCR em paralelo com downscale inteligente (1600px) e warm-up no startup da API:

- **POST** `/api/analisar-rapido-imagem` (Multipart Form)
- **POST** `/api/analisar-rapido` (JSON)

---

## 3. Endpoints Locais / Offline (`digital-document-checker`)

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

## 5. Endpoints Leitura PaddleOCR & Feedback

- **POST** `/api/paddleocr-crlv-imagem` (Multipart Form)
- **POST** `/api/paddleocr-crlv` (JSON)
- **POST** `/api/paddleocr-feedback` (JSON de correção manual para retreino)

Exemplo de Feedback Manual:
```json
POST /api/paddleocr-feedback
{
  "id_amostra": "20261124_135322_90b90f40",
  "placa_correta": "ABC1D23",
  "renavam_correto": "00381050920",
  "observacoes": "Troca de O por 0"
}
```

---

## 6. Treinamento e Retreino Local (Active Learning)

O sistema aprende com fotos adicionadas e feedbacks registrados sem necessidade de marcação manual em imagem:

```bash
# 1. Gera dataset (recorta bboxes e aplica anotações de fotos e feedbacks)
uv run python treinamento/preparar_dataset.py

# 2. Treina a rede CRNN PyTorch e converte para ONNX
uv run python treinamento/treinar_modelo.py
```

O modelo `treinamento/modelos/crnn_crlv.onnx` gerado é carregado automaticamente pela API.

