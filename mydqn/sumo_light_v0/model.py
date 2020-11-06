import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, outputs)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
