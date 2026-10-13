"""PoseNet: 42 -> 100 -> 100 -> 4800 MLP (architecture from the paper)."""
import torch
import torch.nn as nn
import config as C


class PoseNet(nn.Module):
    def __init__(self, in_dim=C.INPUT_DIM, hidden=C.HIDDEN, n_classes=C.NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),  # logits; softmax is inside the loss / at inference
        )

    def forward(self, x):
        return self.net(x)


def load_model(path=C.MODEL_PATH, device="cpu"):
    m = PoseNet().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m
