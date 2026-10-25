import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 4, output_size = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.fc1(x)
        h = self.relu(h)
        z = self.fc2(h)
        y = self.sigmoid(z)
        return y