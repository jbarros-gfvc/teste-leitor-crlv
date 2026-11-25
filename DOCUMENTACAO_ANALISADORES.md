# Documentação dos Analisadores de Documentos (CRLV, CNH, CIN)

Este projeto unifica 4 analisadores distintos de documentos em uma única API FastAPI, cobrindo validação remota oficial, validação criptográfica offline, OCR geométrico e OCR treinado localmente.

---

## 1. Onde os Analisadores se Encontram (Ponto Central)

O ponto de convergência de todos os analisadores é o arquivo [src/teste_leitor_crlv/app.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/app.py).

### Arquitetura de Unificação
```mermaid
graph TD
    Client[Cliente / Insomnia / Swagger] --> App[app.py - FastAPI]
    App --> QRUtils[qrcode_utils.py]
    App --> ModVio[analise_vio.py - Serpro Remoto]
    App --> ModLocal[analise_local.py - Digital Document Checker]
    App --> ModEasy[analise_ocr.py - EasyOCR]
    App --> ModPaddle[analise_paddleocr.py - PaddleOCR ONNX]
    ModPaddle --> RecLocal[reconhecedor_local.py - Modelo Treinado]
    RecLocal --> ModeloWeights[(treinamento/modelos/crnn_crlv.onnx)]
```

- **Entrypoint**: [app.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/app.py) inicializa a aplicação FastAPI e registra os 4 roteadores (`APIRouter`).
- **Enquadramento e Alinhamento**: [enquadramento.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/enquadramento.py) recorta a área útil do cartão antes do OCR através de âncoras geométricas (QR Code à direita, texto Vio e faixa superior escura).
- **Utilitário Compartilhado**: [qrcode_utils.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/qrcode_utils.py) provê `extrair_bytes_qrcode` (`zxing-cpp`) e `gerar_qrcode_base64` (`segno`).
- **Endpoint Raiz de Descoberta**: `GET http://localhost:8000/` retorna o status e catálogo de todos os modos disponíveis.
- **Documentação Interativa**: Swagger UI em `http://localhost:8000/docs` e OpenAPI JSON em `http://localhost:8000/openapi.json`.
- **Coleção de Testes Pronta**: [insomnia_collection.json](file:///C:/Users/user/Documents/teste-leitor-crlv/insomnia_collection.json) contém requisições configuradas para todos os analisadores.

---

## 2. Analisadores por Tipo e Responsabilidade

### Analisador 1: VIO Remoto (API Serpro)
- **Arquivo**: [src/teste_leitor_crlv/analise_vio.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/analise_vio.py)
- **Biblioteca Base**: `vio-api`
- **Tipo de Validação**: Remota (online) via consulta oficial na nuvem Serpro/SENATRAN com chave de acesso.
- **Funcionamento**:
  1. Extrai bytes binários brutos do QR Code na imagem via `zxingcpp`.
  2. Reconstrói a imagem do QR Code em Base64 via `segno`.
  3. Envia para a API Serpro via cliente `VioClient()`.
- **Endpoints**:
  - `POST /api/consultar-imagem` (`multipart/form-data`: campo `file`, flags `wait` e `incluir_qrcode_base64`)
  - `POST /api/consultar` (JSON: `imagem_base64`, `caminho_imagem` ou `texto`)

---

### Analisador 2: Validador Local Offline (Digital Document Checker)
- **Arquivo**: [src/teste_leitor_crlv/analise_local.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/analise_local.py)
- **Biblioteca Base**: `digital-document-checker`
- **Tipo de Validação**: Local, offline (custo zero de API, sem envio de dados para rede).
- **Funcionamento**:
  1. Extrai bytes brutos do QR Code da imagem.
  2. Decodifica assinaturas digitais (ECDSA / RSA / JWS) usando as chaves públicas oficiais embutidas dos órgãos emissores (SENATRAN, SERPRO, MJSP).
  3. Decodifica dados biométricos, dados do condutor e do veículo.
- **Endpoints**:
  - `POST /api/validar-localmente-imagem` (`multipart/form-data`: campo `file`, flag `include_raw`)
  - `POST /api/validar-localmente` (JSON: `imagem_base64`, `caminho_imagem` ou `qrcode_base64`)

---

### Analisador 3: OCR Geométrico (EasyOCR + PyMuPDF)
- **Arquivo**: [src/teste_leitor_crlv/analise_ocr.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/analise_ocr.py)
- **Bibliotecas Base**: `easyocr`, `pymupdf`, `Pillow`, `opencv-python-headless`
- **Tipo de Validação**: Leitura óptica de documento físico/digital (CRLV-e em PDF ou foto/scan).
- **Recursos Principais**:
  - **Correção EXIF**: `ImageOps.exif_transpose` para fotos de celular com metadados de orientação.
  - **Auto-Orientação**: `corrigir_orientacao` via rotação OpenCV (90° e 270°) se faltar PLACA/RENAVAM.
  - **Crop Dinâmico**: `delimitar_documento_pelos_textos` ignora mesa/fundo e calcula dimensões relativas do documento.
  - **6 Âncoras Relativas**: República Federativa, Certificado de Registro, RENAVAM, Placa, Exercício, Número do CRV.
  - **Autocorreção de Caracteres**: `corrigir_placa` converte confusões de OCR (O↔0, I↔1, Z↔2, S↔5, B↔8) segundo a gramática de placas Mercosul/antigas.
- **Endpoints**:
  - `POST /api/ocr-crlv-imagem` (`multipart/form-data`: campo `file`, `margem_erro`)
  - `POST /api/ocr-crlv` (JSON: `caminho_arquivo`, `imagem_base64`, `margem_erro`)

---

### Analisador 4: PaddleOCR + Aprendizado Local ONNX
- **Arquivos**:
  - Analisador: [src/teste_leitor_crlv/analise_paddleocr.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/analise_paddleocr.py)
  - Reconhecedor Local: [src/teste_leitor_crlv/reconhecedor_local.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/reconhecedor_local.py)
  - Pipeline de Treino: [treinamento/](file:///C:/Users/user/Documents/teste-leitor-crlv/treinamento)
- **Bibliotecas Base**: `paddleocr`, `paddlepaddle`, `onnxruntime`, `torch`
- **Tipo de Validação**: Leitura OCR com PP-OCRv6 em runtime ONNX e priorização automática de pesos treinados localmente.
- **Recursos Principais**:
  - **Priorização Automática**: detecta se [treinamento/modelos/crnn_crlv.onnx](file:///C:/Users/user/Documents/teste-leitor-crlv/treinamento/modelos/crnn_crlv.onnx) existe e ativa o motor `local_treinado`. Caso ausente, utiliza o modelo base automaticamente.
  - **Active Learning**: salva fotos com falha de validação em `dados_treinamento/falhas/` gerando um `id_amostra_salva` para composição contínua do dataset.
  - **Feedback do Usuário**: endpoint dedicado para registrar placa e RENAVAM corretos.
  - **Treinamento e Fine-Tuning**: scripts [treinamento/preparar_dataset.py](file:///C:/Users/user/Documents/teste-leitor-crlv/treinamento/preparar_dataset.py) e [treinamento/treinar_modelo.py](file:///C:/Users/user/Documents/teste-leitor-crlv/treinamento/treinar_modelo.py) permitem treinar a rede CRNN com novas fotos a qualquer momento.
- **Endpoints**:
  - `POST /api/paddleocr-crlv-imagem` (`multipart/form-data`: `file`, `margem_erro`, `salvar_amostra`, `usar_modelo_local`)
  - `POST /api/paddleocr-crlv` (JSON: `caminho_arquivo`, `imagem_base64`, `margem_erro`, `usar_modelo_local`)
  - `POST /api/paddleocr-feedback` (JSON: `id_amostra`, `placa_correta`, `renavam_correto`, `observacoes`)

---

### Módulo de Enquadramento Inteligente (`enquadramento.py`)

- **Arquivo**: [src/teste_leitor_crlv/enquadramento.py](file:///C:/Users/user/Documents/teste-leitor-crlv/src/teste_leitor_crlv/enquadramento.py)
- **Objetivo**: Reduzir em até 60% a área de imagem processada pelos modelos OCR cortando mesas, dedos e fundos, e aplicando deskew automático (0°, 90°, 180°, 270°).
- **Âncoras Utilizadas**:
  1. **QR Code VIO (Direita)**: `zxing-cpp` detecta os cantos e o ângulo exato do QR Code. Como os dados do veículo ficam sempre à esquerda e a faixa escura acima, o módulo calcula a ROI exata do cartão com margem de segurança.
  2. **Faixa Superior Escura / Brasão**: Alinha o topo e o limite esquerdo do cabeçalho (*REPÚBLICA FEDERATIVA DO BRASIL*).
  3. **Contorno de Documento**: Caso o QR Code esteja ausente ou rasurado, o detector de bordas Canny com aproximação poligonal localiza o retângulo do papel.
- **Saída nos JSONs da API**:
  ```json
  "enquadramento": {
    "metodo": "qrcode_vio",
    "rotacao_graus": 90,
    "crop_aplicado": true,
    "dimensoes_originais": {"largura": 720, "altura": 1280},
    "dimensoes_finais": {"largura": 981, "altura": 685},
    "qrcode_detectado": true
  }
  ```

---

## 3. Matriz Comparativa dos Analisadores

| Recurso | 1. VIO Remoto | 2. Local Checker | 3. EasyOCR | 4. PaddleOCR + Local |
| :--- | :--- | :--- | :--- | :--- |
| **Ambiente de Execução** | Nuvem Serpro | 100% Local | 100% Local | 100% Local |
| **Custo de Chamada** | Sim (Créditos Serpro) | Gratuito | Gratuito | Gratuito |
| **Dependência de Conexão** | Sim | Não | Não | Não |
| **Entrada Principal** | QR Code VIO | QR Code VIO/Gov | Texto/Foto CRLV | Texto/Foto CRLV |
| **Suporte a PDF e Fotos** | Apenas QR Code | Apenas QR Code | PDF e Fotos | PDF e Fotos |
| **Tolerância a Ângulo** | Média | Média | Alta (cv2 rotate) | Muito Alta (PP-LCNet) |
| **Treinabilidade Local** | Impossível | Chaves Criptográficas | Complexa | Nativa (CRNN/ONNX) |

---

## 4. Guia Completo: Retroalimentação, Feedback e Treinamento

O pipeline de PaddleOCR + CRNN possui arquitetura de **Active Learning** (aprendizado contínuo semi-supervisionado).

### Como Funciona o Ciclo de Retroalimentação

```mermaid
flowchart TD
    Doc["Documento Enviado à API"] --> Det["1. Detecção Textual (DBNet ONNX)"]
    Det --> Rec["2. Reconhecimento (CRNN Local ONNX)"]
    Rec --> Val{"3. Validação de Regras (Placa + RENAVAM)"}
    
    Val -- "Válido" --> RespOK["Resposta com Sucesso"]
    Val -- "Falha de Leitura" --> SaveFail["4. Salva Imagem em dados_treinamento/falhas/<br/>Gera id_amostra"]
    
    SaveFail --> UserFeedback["5. Usuário envia POST /api/paddleocr-feedback<br/>(com placa e renavam corretos)"]
    UserFeedback --> JSONUpdate["Atualiza dados_treinamento/falhas/{id_amostra}.json"]
    
    JSONUpdate --> Prep["6. Executa preparar_dataset.py<br/>(Corta bboxes e aplica anotações corrigidas)"]
    Prep --> Train["7. Executa treinar_modelo.py<br/>(Treina PyTorch e exporta ONNX)"]
    Train --> NewWeights["8. Novo crnn_crlv.onnx ativo imediatamente"]
```

### 1. Envio de Feedback Manual

> [!NOTE]
> **Não é necessário marcar nem desenhar na imagem.**
> O detector DBNet já calcula as coordenadas das caixas de texto. O feedback manual serve exclusivamente para fornecer o texto correto caso o OCR tenha confundido caracteres.

Quando uma imagem falha na validação, a API retorna o campo `id_amostra_salva`:

```json
{
  "sucesso": false,
  "modo": "leitura_documentos_paddleocr",
  "analise": {
    "formato_crlv_valido": false,
    "placa": null,
    "renavam": "00381050920",
    "id_amostra_salva": "20261124_135322_90b90f40"
  }
}
```

Para enviar a correção, faça uma requisição `POST /api/paddleocr-feedback`:

- **Endpoint**: `POST http://localhost:8000/api/paddleocr-feedback`
- **Headers**: `Content-Type: application/json`
- **Body**:

```json
{
  "id_amostra": "20261124_135322_90b90f40",
  "placa_correta": "ABC1D23",
  "renavam_correto": "00381050920",
  "observacoes": "Troca de O por 0 na placa"
}
```

- **Resposta da API**:
```json
{
  "sucesso": true,
  "mensagem": "Feedback gravado com sucesso para retreino."
}
```

---

### 2. Adicionando Novas Imagens para Treinamento

Além das falhas capturadas pela API, você pode incluir novos lotes de fotos a qualquer momento:

1. Coloque fotos de CRLV (JPG/PNG) na pasta `imgs/`.
2. As imagens serão automaticamente processadas e recortadas no próximo passo.

---

### 3. Passo a Passo de Execução do Treinamento

#### Passo 1: Preparar o Dataset
Lê a pasta `imgs/` e a pasta `dados_treinamento/falhas/` (aplicando correções manuais se houverem), recorta cada linha de texto e gera o arquivo `treinamento/dataset/labels.txt`:

```bash
uv run python treinamento/preparar_dataset.py
```

*Saída esperada:*
```text
[1/3] Inicializando PaddleOCR...
[2/3] Processando 18 imagens de amostra em 'imgs'...
  -> [1/18] foto1.jpg: 17 recortes gerados
[2.1/3] Processando amostras ativas em 'dados_treinamento/falhas'...
  -> Falha [1/2] 20261124_135322_90b90f40.jpg: 8 recortes (com feedback manual)
[3/3] Concluído! Total de 317 recortes salvos em 'treinamento/dataset/images'.
```

#### Passo 2: Treinar a Rede CRNN e Exportar ONNX
Treina a rede neural convolucional-recorrente (PyTorch) usando CTC Loss, salva os pesos `.pth` e converte automaticamente para `.onnx`:

```bash
uv run python treinamento/treinar_modelo.py
```

*Parâmetros padrão:*
- **Épocas**: 20 (ajustável no script)
- **Batch Size**: 16
- **Learning Rate**: 0.0005 com otimizador AdamW
- **Saídas geradas**:
  - `treinamento/modelos/crnn_crlv.pth` (pesos PyTorch)
  - `treinamento/modelos/crnn_crlv.onnx` (modelo de inferência de alta velocidade)
  - `treinamento/modelos/vocabulario.json` (dicionário de caracteres mapeados)

---

### 4. Como a API Passa a Usar o Novo Modelo

A transição é **100% transparente**:

1. A API verifica a existência do arquivo `treinamento/modelos/crnn_crlv.onnx`.
2. Assim que o arquivo for atualizado pelo comando de treino, o `ReconhecedorCRNNLocal` passa a inferir com os novos pesos nas próximas requisições.
3. Se o arquivo `.onnx` for apagado ou movido, a API volta automaticamente a utilizar o modelo base global do PaddleOCR (fallback seguro sem quebrar o sistema).

---

## 5. Como Executar a API

```bash
# Executar a API via script registrado
uv run teste-leitor-crlv

# Ou diretamente via Uvicorn
uv run uvicorn teste_leitor_crlv.app:app --reload --port 8000
```
