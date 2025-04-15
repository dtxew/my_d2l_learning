import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torchvision import transforms
from torch.utils import data

trans=transforms.ToTensor()#把图片转为tensor,transforms=trans
batch_size=256

#训练数据集
mnist_train=torchvision.datasets.FashionMNIST("ch4/data",train=True
                                             ,transform=trans,download=True)
#测试数据集
mnist_test=torchvision.datasets.FashionMNIST("ch4/data",train=False,transform=trans,
                                             download=True)

#获取数据迭代器
train_iter=data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=4)

test_iter=data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=4)

net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
# 定义了一个顺序容器，包含两个层：Flatten层将输入展平，Linear层是一个全连接层，将784维的输入映射到10维的输出

def init_weights(m):
    # 如果模块是线性层，则使用标准差为0.01的正态分布初始化权重
    if(type(m)==nn.Linear):
        nn.init.normal_(m.weight,std=0.01)

# 应用权重初始化函数到网络的每个模块
net.apply(init_weights)

loss=nn.CrossEntropyLoss()

trainer=torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs=10

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net,train_iter,loss,updater):
    net.train()
    #损失总和，样本数总和，样本数
    metric=d2l.Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        updater.zero_grad()
        l.mean().backward()# 计算张量 l 的平均值并对该平均值进行反向传播，计算梯度
        updater.step()# 执行优化器的更新步骤
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_main(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics=train_epoch(net,train_iter,loss,updater)
        acc=evaluate_accuracy(net,test_iter)
        train_loss,train_acc=train_metrics
        print(f'Epoch {epoch + 1}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test acc: {acc:.4f}')

if __name__=='__main__':
    train_main(net,train_iter,test_iter,loss,3,trainer)