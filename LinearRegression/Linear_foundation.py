# 从零开始实现线性回归模型

import random
import torch


# 生成数据
def synthetic_data(w, b, num_examples):
    '''生成 y = wx +b + 噪声'''
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)

    return x, y.reshape((-1, 1))  # 作为列向量返回

true_w = torch.tensor([2, -3.4])
true_b = 4.2
feature, labels = synthetic_data(true_w, true_b, 1000)

# print('features:', feature[0], '\nlebals:', labels[0])

# 按照batch_size 拿数据
def batch_iter(batch_size, feature, labels):
    num_examples = len(labels)
    indices = list(range(num_examples))

    # 将样本读取的顺序打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, num_examples)]
        )
        yield feature[batch_indices], labels[batch_indices]

batch_size = 10

# for x, y in batch_iter(batch_size, feature, labels):
#     print(x, '\n', y)
#     break

# 权重和偏差
w = torch.normal(0, 0.1, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# 线性回归模型
def Linear(w, x, b):
    return torch.matmul(x, w) + b

# 均方误差
def squared_loss(y_pre, y):
    return (y_pre-y.reshape(y_pre.shape))**2/2

# 小批量随机梯度下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            # 损失函数中没有求均值
            param -= lr * param.grad/batch_size
            # 梯度清零
            param.grad.zero_()

# 训练过程

lr = 0.03
num_epochs = 3
loss = squared_loss
model = Linear


for epoch in range(num_epochs):
    for x, y in batch_iter(batch_size, feature, labels):
        l = loss(model(w, x, b), y) # Loss: (10,1)
        l.sum().backward()
        sgd([w,b], lr, batch_size) # 梯度更新

    with torch.no_grad():
        # 计算当前
        train_1 = loss(model(w, feature, b), labels)
        print(f'{epoch+1}/{num_epochs}: ------ loss: {float(train_1.mean()):.6f}')



