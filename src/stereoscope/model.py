import torch
from torch import nn


class ConfidenceCNN(nn.Module):
    """Local 7x7 receptive field; predicts disparity-error logit, not disparity."""
    def __init__(self, channels=6):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class LinearConfidence(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.net = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


def choose_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
