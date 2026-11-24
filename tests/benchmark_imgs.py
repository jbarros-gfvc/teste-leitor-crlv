import os

from teste_leitor_crlv.analise_ocr import analisar_crlv_com_posicao
from teste_leitor_crlv.analise_paddleocr import analisar_crlv_com_paddleocr

imgs_dir = "imgs"
files = [f for f in os.listdir(imgs_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(
    f"{'ARQUIVO':20} | {'PADDLE OK':9} | {'PLACA PADDLE':12} | {'RENAVAM PADDLE':14} | {'EASYOCR OK':10} | {'PLACA EASY':12}"
)
print("-" * 90)

for f in sorted(files):
    p = os.path.join(imgs_dir, f)
    # PaddleOCR
    try:
        p_res = analisar_crlv_com_paddleocr(p)
        p_ok = str(p_res["formato_crlv_valido"])
        p_placa = str(p_res["placa"])
        p_renavam = str(p_res["renavam"])
    except Exception as e:  # noqa: BLE001
        p_ok = "ERR"
        p_placa = str(e)[:10]
        p_renavam = ""

    # EasyOCR
    try:
        e_res = analisar_crlv_com_posicao(p)
        e_ok = str(e_res["formato_crlv_valido"])
        e_placa = str(e_res["placa"])
    except Exception as e:  # noqa: BLE001
        e_ok = "ERR"
        e_placa = str(e)[:10]

    print(f"{f:20} | {p_ok:9} | {p_placa:12} | {p_renavam:14} | {e_ok:10} | {e_placa:12}")
