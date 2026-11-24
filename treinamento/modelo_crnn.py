import torch
from torch import nn


class CRNN(nn.Module):
    """
    Arquitetura CRNN padrão (CNN + BiLSTM + Projeção Linear) para reconhecimento de texto.
    Entrada: Imagem monocromática (B, 1, 32, W)
    Saída: Log-probabilidades (T, B, num_classes) para CTCLoss
    """

    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.num_classes = num_classes

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 16 x W/2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 8 x W/4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # -> 4 x W/4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # -> 2 x W/4
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # -> 1 x W/4
        )

        # RNN Sequence Modeling
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )

        # Classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, W)
        conv = self.cnn(x)  # (B, 512, 1, W_seq)
        _b, _c, h, _w = conv.size()
        assert h == 1, f"Altura intermediária esperada 1, recebido {h}"

        conv = conv.squeeze(2)  # (B, 512, W_seq)
        conv = conv.permute(2, 0, 1)  # (W_seq, B, 512)

        recurrent, _ = self.rnn(conv)  # (W_seq, B, 512)
        out = self.fc(recurrent)  # (W_seq, B, num_classes)
        return self.log_softmax(out)
