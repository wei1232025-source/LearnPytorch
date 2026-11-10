import torch.nn as nn
import torch
import torch.optim as optim

from model import simple_model


torch .manual_seed(42)

# 构造带有噪声的非线性数据
N = 300
x = torch.linspace(-3, 3, N).reshape(-1, 1)
y = torch.sin(2*x) + 0.25*torch.randn(N, 1)

perm = torch.randperm(N)
train_index = perm[:210]
val_index = perm[210: ]

x_train, y_train = x[train_index], y[train_index]
x_val, y_val = x[val_index], y[val_index]

# 训练与验证
def run_experiment(weight_decay):
    model = simple_model()
    optimizer = optim.AdamW(
        model.parameters(),
        lr = 0.01,
        weight_decay = weight_decay
    )
    loss_fn = nn.MSELoss()

    for epoch in range(301):
        model.train()

        pre = model(x_train)
        loss = loss_fn(pre, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%100 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = loss_fn(
                    model(x_train), y_train
                ).item()

                val_loss = loss_fn(
                    model(x_val), y_val
                ).item()

                print(
                    f"weight_decay = {weight_decay}, "
                    f"epoch = {epoch}, "
                    f"train_loss = {train_loss:.4f}, "
                    f"val_loss = {val_loss:.4f}"
                )


run_experiment(0.0)
run_experiment(0.1)


