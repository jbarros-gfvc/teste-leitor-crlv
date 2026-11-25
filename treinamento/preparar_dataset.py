import json
import os
from pathlib import Path

import numpy as np

# Evita conflitos de DLL no Windows
from paddleocr import PaddleOCR
from PIL import Image

DIR_RAIZ = Path(__file__).resolve().parent
DIR_IMGS = Path("imgs")
DIR_FALHAS = Path("dados_treinamento/falhas")
DIR_DATASET = DIR_RAIZ / "dataset"
DIR_CROPS = DIR_DATASET / "images"
DIR_CROPS.mkdir(parents=True, exist_ok=True)


def recortar_e_anotar(pil_img, ocr, prefixo_crop: str, feedback: dict | None = None) -> list[str]:
    largura, altura = pil_img.size
    img_np = np.array(pil_img)

    out = ocr.predict(img_np)
    if not out or len(out) == 0:
        return []

    pagina = out[0]
    polys = pagina.get("rec_polys", [])
    texts = pagina.get("rec_texts", [])
    scores = pagina.get("rec_scores", [])

    linhas = []
    for p_idx, (poly, text, score) in enumerate(zip(polys, texts, scores)):
        texto_limpo = str(text).strip().upper()

        # Se houver feedback do usuário, substitui anotações em alvos chave
        if feedback:
            if feedback.get("placa_correta") and "PLACA" in texto_limpo:
                texto_limpo = feedback["placa_correta"].upper()
            elif feedback.get("renavam_correto") and len(texto_limpo) == 11 and texto_limpo.isdigit():
                texto_limpo = feedback["renavam_correto"]

        if len(texto_limpo) < 2 or float(score) < 0.60:
            continue

        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        x_min = max(0, int(min(xs)))
        y_min = max(0, int(min(ys)))
        x_max = min(largura, int(max(xs)))
        y_max = min(altura, int(max(ys)))

        if (x_max - x_min) < 10 or (y_max - y_min) < 8:
            continue

        recorte = pil_img.crop((x_min, y_min, x_max, y_max))
        nome_crop = f"{prefixo_crop}_{p_idx:02d}.jpg"
        caminho_crop = DIR_CROPS / nome_crop
        recorte.save(caminho_crop, quality=95)

        linhas.append(f"images/{nome_crop}\t{texto_limpo}\n")

    return linhas


def preparar():
    print("[1/3] Inicializando PaddleOCR...")
    ocr = PaddleOCR(lang="pt", engine="onnxruntime")

    amostras_geradas = 0
    linhas_anotacao = []

    # 1. Processar pasta imgs/ se existir
    if DIR_IMGS.exists():
        arquivos = [f for f in os.listdir(DIR_IMGS) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"[2/3] Processando {len(arquivos)} imagens de amostra em '{DIR_IMGS}'...")
        for idx_arq, arq in enumerate(sorted(arquivos), 1):
            caminho_img = DIR_IMGS / arq
            try:
                pil_img = Image.open(caminho_img).convert("RGB")
                novas_linhas = recortar_e_anotar(pil_img, ocr, f"crop_img_{idx_arq:02d}")
                linhas_anotacao.extend(novas_linhas)
                amostras_geradas += len(novas_linhas)
                print(f"  -> [{idx_arq}/{len(arquivos)}] {arq}: {len(novas_linhas)} recortes gerados")
            except Exception as exc:  # noqa: BLE001
                print(f"  -> [{idx_arq}/{len(arquivos)}] {arq}: Erro ({exc})")

    # 2. Processar amostras de falhas registradas com ou sem feedback manual
    if DIR_FALHAS.exists():
        amostras_falhas = [f for f in os.listdir(DIR_FALHAS) if f.lower().endswith(".jpg")]
        if amostras_falhas:
            print(f"[2.1/3] Processando {len(amostras_falhas)} amostras ativas em '{DIR_FALHAS}'...")
            for idx_f, arq_f in enumerate(sorted(amostras_falhas), 1):
                caminho_img = DIR_FALHAS / arq_f
                json_path = DIR_FALHAS / f"{Path(arq_f).stem}.json"
                feedback = None
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as jf:
                            d = json.load(jf)
                            feedback = d.get("feedback_usuario")
                    except (OSError, ValueError):
                        feedback = None

                try:
                    pil_img = Image.open(caminho_img).convert("RGB")
                    novas_linhas = recortar_e_anotar(pil_img, ocr, f"crop_falha_{idx_f:02d}", feedback=feedback)
                    linhas_anotacao.extend(novas_linhas)
                    amostras_geradas += len(novas_linhas)
                    fb_str = " (com feedback manual)" if feedback else ""
                    print(f"  -> Falha [{idx_f}/{len(amostras_falhas)}] {arq_f}: {len(novas_linhas)} recortes{fb_str}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  -> Falha [{idx_f}/{len(amostras_falhas)}] {arq_f}: Erro ({exc})")

    arquivo_labels = DIR_DATASET / "labels.txt"
    with open(arquivo_labels, "w", encoding="utf-8") as f:
        f.writelines(linhas_anotacao)

    print(f"[3/3] Concluído! Total de {amostras_geradas} recortes salvos em '{DIR_CROPS}'.")
    print(f"      Arquivo de anotações: '{arquivo_labels}'")


if __name__ == "__main__":
    preparar()
