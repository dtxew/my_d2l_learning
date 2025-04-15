import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w=torch.tensor([2,-3.4])
true_b=4.2

features,labels=d2l.synthetic_data(true_w,true_b,1000)

def load_array(data_arrays, batch_size, is_train=True):
    # 将数据数组转换为TensorDataset，以便于后续的数据加载
    dataset = data.TensorDataset(*data_arrays) #星号为解包操作，将列表解开为每个单独变量
    # 创建DataLoader实例，根据is_train参数决定是否打乱数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size=10

data_iter=load_array((features,labels),batch_size)

#定义模型-线性模型

net=nn.Sequential(nn.Linear(2,1))
# 定义了一个顺序容器(list)，包含一个线性层，该层将输入的2维特征映射到1维输出

# 初始化网络的第一层权重为均值为0，标准差为0.01的正态分布
net[0].weight.data.normal_(0, 0.01)
# 初始化网络的第一层偏置为0
net[0].bias.data.fill_(0)

loss=nn.MSELoss()
# 定义优化器，使用随机梯度下降（SGD）算法，学习率为0.03
trainer=torch.optim.SGD(net.parameters(),lr=0.03)#第一个是神经网络中的参数(w,b)

epochs=3

for epoch in range(epochs):
    for x,y in data_iter:
        l=loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step() # 执行训练步骤，更新模型参数
  
    l=loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
# 此循环用于训练神经网络模型，遍历多个epoch和每个epoch中的数据批次
# 计算每个批次的损失，并更新模型参数
# 在每个epoch结束后，计算在整个数据集上的损失并打印出来

