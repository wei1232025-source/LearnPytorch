import torch
from model_mlp import MLP
import torch.nn as nn

torch.manual_seed(42)

x = torch.tensor([
    [0., 1.],
    [3., 1.],
    [1., 2.],
    [4., 4.]
])
y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

model = MLP(2, 4, 1)
optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.05)
Loss = nn.BCELoss()

epochs = 1000

for epoch in range(epochs):
    p = model(x)

    L = Loss(p, y)

    optimizer.zero_grad()

    L.backward()

    optimizer.step()

    if((epoch+1)%100 == 0):
        print(f"[{epoch+1}]/[{epochs}]----Loss: {L:.5f}")
