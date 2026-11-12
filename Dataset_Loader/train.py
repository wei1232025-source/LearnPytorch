import torch
import torch.nn as nn
from model import MLP
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)
N = 2000

# 构建数据集

# 生成随机分类标签: 0或1
y = torch.randint(0, 2, (N,))
# print(y.shape)   torch.Size([2000])

# 根据类别生成不同的半径
r = torch.where(
    y == 0,
    0.6 + 0.25 * torch.rand(N),
    1.5 + 0.25 * torch.rand(N)
)

# 随机生成角度
theta = 2*torch.pi*torch.rand(N)

# 二维坐标
X = torch.stack([
    r*torch.cos(theta),
    r*torch.sin(theta)
    ],
    dim = 1
)
# print(X.shape)  torch.Size([2000, 2])

# 添加少量噪声
X = X + 0.05*torch.rand_like(X)
# 将标签转化为浮点数
y = y.float()



dataset = TensorDataset(X, y)

train_dataset, val_dataset = random_split(
    dataset, [1600, 400], generator=torch.Generator().manual_seed(42)
)

# 构建Dataloder
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size = 64,
    shuffle=False
)

# 模型、损失函数、优化器
model = MLP()
Loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.001
)

epochs = 30
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_trainloss = 0.0
    train_correct = 0

    for x1, y1 in train_loader:

        optimizer.zero_grad()

        logits = model(x1)

        loss = Loss(logits, y1)

        loss.backward()

        optimizer.step()

        # 统计训练效果
        total_trainloss += loss.item() * len(y1)
        predictions = (logits>=0).float()
        train_correct += (predictions==y1).sum().item()

    train_loss = total_trainloss / len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    # 验证阶段
    model.eval()

    val_loss_total = 0.0
    val_correct = 0

    with torch.no_grad():
        for xb, yb in val_loader:

            logits = model(xb)
            loss = Loss(logits, yb)

            val_loss_total += loss.item() * len(yb)

            predictions = (logits >= 0).float()
            val_correct += (predictions == yb).sum().item()

    val_loss = val_loss_total / len(val_dataset)
    val_acc = val_correct / len(val_dataset)

    # ---------- 保存最佳模型 ----------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2%}"
    )

print("训练完成！")





    




