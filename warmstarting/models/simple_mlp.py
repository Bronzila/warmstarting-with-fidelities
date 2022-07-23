import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
