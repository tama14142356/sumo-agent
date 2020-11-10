import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
