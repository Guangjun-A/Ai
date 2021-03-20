# -*- coding = utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
'''框架流程'''
# 1.数据处理，加载数据
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


# 1.1 创建一个 Dataset 对象。必须实现__len__()、getitem()这两个方法，这里面会用到transform对数据集进行扩充。
class Process_data(Dataset):
    def __init__(self):
        super(Process_data, self).__init__()
        self.data = None  # 真实数据
        self.image = []
        self.lable = []
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return (self.image[item], self.lable[item])


dataset = Process_data()
train_samples = int(len(dataset) * 0.8)
test_samples = len(dataset) - train_samples
train_set, test_set = torch.utils.data.random_split(dataset=dataset, lengths=[train_samples, test_samples])

# 1.2 创建一个 DataLoader 对象
train_loader = DataLoader(dataset=train_set,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4)
test_loader = DataLoader(dataset=train_set,
                         batch_size=4,
                         shuffle=True,
                         num_workers=4)


# 2. 搭建模型， 实例化模型
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        # 搭建网络
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 搭建网络

    def forward(self, x):
        return x

model = Model(in_channels=3, out_channels=64, kernel_size=1)

# 3. 定义损失函数与优化器
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.1,
                             weight_decay=0.001)

# 4. 训练模型
num_epochs = 1000
mode = "train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 4.1 训练阶段
if mode == "train":
    model.train()  # 设置模型为训练状态
    train = train_loader  # 加载数据集
    loss_value = []
    n_total_steps = len(train)
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(train):
            # 得到数据
            images = image.float().to(device)
            labels = label.long().to(device)

            # 前向传播，求loss
            output = model(images)
            loss = loss_func(output, labels)

            # 反向传播，最小化loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value.append(loss.data.item())

            # 打印输出
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
 # 4.1 测试阶段
elif mode == "test":
    evaluation = True
    model.eval()  # 设置模型为测试（评估）阶段
    test = test_loader  # 加载数据集
    result_frag = []
    loss_value = []
    label_frag = []

    for image, label in test:
        # 得到数据
        images = image.float().to(device)
        labels = label.long().to(device)

        # 预测，得到推断值
        with torch.no_grad():
            output = model(images)
        result_frag.append(output.data.cpu().numpy())

        # 获得损失函数
        if evaluation == True:
            loss = loss_func(output, label)
            loss_value.append(loss.item())
            label_frag.append(label.data.cpu().numpy())
    result = np.concatenate(result_frag)
    if evaluation:
        label = np.concatenate(label_frag)
        # 显示top_k的准确率
        for k in [1, 5]:
            rank = result.argsort()
            hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
            accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
