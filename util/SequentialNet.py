import torch
from torch import nn
import torch.nn.functional as F

class SequentialNet(nn.Module):
    """A sequential model you can specify input, output, hidden size and depth for
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128, depth: int = 3):
        super(SequentialNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size if depth > 0 else output_size))

        for _ in range(1, depth):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))

        if depth > 0:
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, output_size))

        self.net = nn.Sequential(*layers)

        self.seed = torch.manual_seed(42)

    def forward(self, x):
        return self.net(x)