from torch.utils import data
import torch
from Linear_foundation import synthetic_data
import torch.nn as nn


true_w = torch.tensor([2, -3.4])
true_b = 4.2

feature, labels = synthetic_data(true_w, true_b, num_examples=1000)

# 构造一个迭代器
def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

batch_size = 10
data_iter = load_array((feature, labels), batch_size)

next(iter(data_iter))

# model
net = nn.Sequential(nn.Linear(2,1))
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 损失函数
loss = nn.MSELoss()

# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练过程
epochs = 3

for epoch in range(epochs):
    for x, y in data_iter:
        l = loss(net(x), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()

    l = loss(net(feature), labels)
    print(f'{epoch+1}/{epochs}--------loss: {l:.3f}')



