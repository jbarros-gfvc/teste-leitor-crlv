import json
import random
from pathlib import Path

import numpy as np
import torch
from modelo_crnn import CRNN
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

DIR_RAIZ = Path(__file__).resolve().parent
DIR_DATASET = DIR_RAIZ / "dataset"
DIR_MODELOS = DIR_RAIZ / "modelos"
DIR_MODELOS.mkdir(parents=True, exist_ok=True)


class CRLVDataset(Dataset):
    def __init__(self, linhas, vocab_map):
        self.itens = []
        for linha in linhas:
            partes = linha.strip().split("\t")
            if len(partes) == 2:
                rel_path, texto = partes
                img_path = DIR_DATASET / rel_path
                if img_path.exists():
                    self.itens.append((img_path, texto))
        self.vocab_map = vocab_map

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, idx):
        img_path, texto = self.itens[idx]
        img = Image.open(img_path).convert("L")

        # Redimensiona para altura fixa 32 mantendo proporção (largura máx 256)
        w, h = img.size
        nova_w = max(16, min(256, int(w * (32.0 / h))))
        img = img.resize((nova_w, 32), Image.Resampling.BILINEAR)

        # Pad para 256 de largura
        tensor_img = np.zeros((1, 32, 256), dtype=np.float32)
        arr = np.array(img, dtype=np.float32) / 255.0
        # Normalização [-1, 1]
        arr = (arr - 0.5) / 0.5
        tensor_img[0, :, :nova_w] = arr

        rotulos = [self.vocab_map.get(c, self.vocab_map["?"]) for c in texto]
        return torch.tensor(tensor_img, dtype=torch.float32), torch.tensor(rotulos, dtype=torch.long), len(rotulos)


def collate_fn(batch):
    imgs, targets, target_lengths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    # Concatena alvos para CTCLoss
    concatenated_targets = torch.cat(targets, dim=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return imgs, concatenated_targets, target_lengths


def treinar():
    print("[1/4] Carregando dataset e anotações...")
    arquivo_labels = DIR_DATASET / "labels.txt"
    if not arquivo_labels.exists():
        print(f"Erro: '{arquivo_labels}' não encontrado. Execute 'preparar_dataset.py' primeiro!")
        return

    with open(arquivo_labels, "r", encoding="utf-8") as f:
        linhas = [l for l in f if l.strip()]

    print(f"Total de amostras encontradas: {len(linhas)}")

    # Constrói vocabulário único (índice 0 reservado para 'BLANK' do CTC)
    todos_chars = sorted(set("".join([l.split("\t")[1].strip() for l in linhas if "\t" in l])))
    if "?" not in todos_chars:
        todos_chars.append("?")

    # 0 = BLANK
    char_to_idx = {c: i + 1 for i, c in enumerate(todos_chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(todos_chars)}
    idx_to_char[0] = ""
    num_classes = len(todos_chars) + 1

    print(f"Classes no vocabulário: {num_classes} caracteres únicos.")

    # Salva vocabulário
    with open(DIR_MODELOS / "vocabulario.json", "w", encoding="utf-8") as f:
        json.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}, f, ensure_ascii=False, indent=2)

    random.seed(42)
    random.shuffle(linhas)

    split = max(1, int(len(linhas) * 0.85))
    linhas_treino = linhas[:split]
    linhas_val = linhas[split:] if len(linhas) > 1 else linhas

    ds_treino = CRLVDataset(linhas_treino, char_to_idx)
    loader_treino = DataLoader(ds_treino, batch_size=min(8, len(ds_treino)), shuffle=True, collate_fn=collate_fn)

    print("[2/4] Inicializando modelo CRNN (PyTorch)...")
    model = CRNN(num_classes=num_classes)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print("[3/4] Iniciando loop de treinamento (20 épocas)...")
    epochs = 20
    model.train()

    for epoch in range(1, epochs + 1):
        loss_total = 0.0
        batches = 0
        for imgs, targets, target_lengths in loader_treino:
            optimizer.zero_grad()
            # imgs: (B, 1, 32, 256)
            preds = model(imgs)  # (T, B, num_classes)
            T, B, _ = preds.size()

            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
            loss = criterion(preds, targets, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_total += loss.item()
            batches += 1

        loss_media = loss_total / max(1, batches)
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            print(f"  -> Época [{epoch:02d}/{epochs:02d}] - Loss CTC médio: {loss_media:.4f}")

    print("[4/4] Salvando modelo treinado...")
    caminho_pesos = DIR_MODELOS / "crnn_crlv.pth"
    torch.save(model.state_dict(), caminho_pesos)
    print(f"Pesos salvos em: '{caminho_pesos}'")

    # Avaliação rápida
    model.eval()
    print("\n--- Teste de Inferência no Modelo Treinado ---")
    with torch.no_grad():
        ds_val = CRLVDataset(linhas_val[:5], char_to_idx)
        for i in range(min(5, len(ds_val))):
            img_tensor, _rotulos, _ = ds_val[i]
            preds = model(img_tensor.unsqueeze(0))  # (T, 1, num_classes)
            # Greedy decode
            pred_indices = preds.squeeze(1).argmax(dim=-1).cpu().numpy()

            # Collapse CTC repeats and blanks
            chars = []
            prev = None
            for idx in pred_indices:
                if idx != 0 and idx != prev:
                    chars.append(idx_to_char.get(idx, ""))
                prev = idx

            pred_texto = "".join(chars)
            real_texto = ds_val.itens[i][1]
            print(f"  Original: '{real_texto}'  -->  Predição: '{pred_texto}'")

    print("\nTreinamento de teste concluído com sucesso!")


if __name__ == "__main__":
    treinar()
