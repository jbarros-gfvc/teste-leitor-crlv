import os
from pathlib import Path

import numpy as np

# Evita conflitos de DLL no Windows
from paddleocr import PaddleOCR
from PIL import Image

DIR_RAIZ = Path(__file__).resolve().parent
DIR_IMGS = Path("imgs")
DIR_DATASET = DIR_RAIZ / "dataset"
DIR_CROPS = DIR_DATASET / "images"
DIR_CROPS.mkdir(parents=True, exist_ok=True)


def preparar():
    print("[1/3] Inicializando PaddleOCR...")
    ocr = PaddleOCR(lang="pt", engine="onnxruntime")

    arquivos = [f for f in os.listdir(DIR_IMGS) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[2/3] Processando {len(arquivos)} imagens de amostra em '{DIR_IMGS}'...")

    amostras_geradas = 0
    linhas_anotacao = []

    for idx_arq, arq in enumerate(sorted(arquivos), 1):
        caminho_img = DIR_IMGS / arq
        try:
            pil_img = Image.open(caminho_img).convert("RGB")
            largura, altura = pil_img.size
            img_np = np.array(pil_img)

            out = ocr.predict(img_np)
            if not out or len(out) == 0:
                continue

            pagina = out[0]
            polys = pagina.get("rec_polys", [])
            texts = pagina.get("rec_texts", [])
            scores = pagina.get("rec_scores", [])

            for p_idx, (poly, text, score) in enumerate(zip(polys, texts, scores)):
                texto_limpo = str(text).strip().upper()
                if len(texto_limpo) < 2 or float(score) < 0.60:
                    continue

                # Extrai bbox retangular para crop
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                x_min = max(0, int(min(xs)))
                y_min = max(0, int(min(ys)))
                x_max = min(largura, int(max(xs)))
                y_max = min(altura, int(max(ys)))

                if (x_max - x_min) < 10 or (y_max - y_min) < 8:
                    continue

                recorte = pil_img.crop((x_min, y_min, x_max, y_max))

                nome_crop = f"crop_{idx_arq:02d}_{p_idx:02d}.jpg"
                caminho_crop = DIR_CROPS / nome_crop
                recorte.save(caminho_crop, quality=95)

                linhas_anotacao.append(f"images/{nome_crop}\t{texto_limpo}\n")
                amostras_geradas += 1

            print(f"  -> [{idx_arq}/{len(arquivos)}] {arq}: {len(texts)} textos detectados")
        except Exception as exc:  # noqa: BLE001
            print(f"  -> [{idx_arq}/{len(arquivos)}] {arq}: Erro ({exc})")

    arquivo_labels = DIR_DATASET / "labels.txt"
    with open(arquivo_labels, "w", encoding="utf-8") as f:
        f.writelines(linhas_anotacao)

    print(f"[3/3] Concluído! Total de {amostras_geradas} recortes salvos em '{DIR_CROPS}'.")
    print(f"      Arquivo de anotações: '{arquivo_labels}'")


if __name__ == "__main__":
    preparar()
