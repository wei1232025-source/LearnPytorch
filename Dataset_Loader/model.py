import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    # squeeze(1): 删除序号为1的维度，保证输出维度与Y相同
    def forward(self, x):
        # print(x.shape)
        return self.net(x).squeeze(1)
    