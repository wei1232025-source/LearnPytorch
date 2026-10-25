import torch
from model_mlp import MLP

model = MLP()
x1 = torch.tensor([1., 4.])

with torch.no_grad():
    probability = model(x1)

label = (probability>=0.5).int()

print("probability: ", probability.item())
print("label：", label.item())