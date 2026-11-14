import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import simple_cnn



epochs = 30
batch_size = 128
lr = 0.01

# 准备工作

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("当前设备", device)

# 数据加载
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root = "./CNN/data",
    train=True,
    transform=transform,
    download=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
# print(len(dataset)) 60000
# image, label = dataset[0]

# print("图像形状：", image.shape)   torch.Size([1, 28, 28])
# print("图像标签：", label)         5

# plt.imshow(image.squeeze(0), cmap="gray")
# plt.title(f"Label: {label}")
# plt.show()

test_dataset = datasets.MNIST(
    root = "./CNN/data",
    train = False,
    transform=transform,
    download=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
# print(len(test_dataset))  10000



# 损失函数，， 优化器
model = simple_cnn().to(device)
# print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = lr
)

def train():
    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)
            # print(images.device)   cuda
            # 预测值

            y = model(images)


            # 计算损失
            loss = criterion(y, labels)

            #清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            train_loss += loss.item()*images.size(0)

            predictions = y.argmax(dim=1)
            correct += (predictions==labels).sum().item()
            total += labels.size(0)

        avg_loss = train_loss / total
        train_acc = 100*correct / total

        print(f"{epoch+1}/{epochs}----avg_loss : {avg_loss:.4f}  train_acc : {train_acc:.4f}%")
        torch.save(
            model.state_dict(),
            "./CNN/model.pth"
        )

if __name__ == "__main__":
    train()

    
