# MNIST数据集实现手写数字识别


import torch
import torch.nn as nn

class simple_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            # padding=1, kernel_size=3, stride=1 卷积图像大小不变 img_size = [3, 64, 64]

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classfier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 10)
        )

    def forward(self, x):

        # for name, params in self.feature.named_parameters():
        #     print(name)
        #     print(params.device)   cuda
        x = self.feature(x)
        x = self.classfier(x)

        return x

# model = simple_cnn()
# x = torch.rand(32, 3, 64, 64)
# x = model(x)
# print(x.shape)
# for name, param in model.named_parameters():
#     print(name, param.shape)