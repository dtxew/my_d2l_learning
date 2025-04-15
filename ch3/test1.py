import torch
import random
from d2l import torch as d2l

#线性回归

#输入:{x1,x2,...,xn}
#权重:{w1,w2,...,wn}
#f=(xw)+b

#平方损失和梯度下降


#首先设真实的w=[2,-3.4],b=4.2

#y=xw+b+e(随机噪声)
def synthic_data(w,b,nums_example):
    X=torch.normal(0,1,(nums_example,len(w)))
    # 生成一个均值为0，标准差为1的正态分布随机数张量，形状为(nums_example, len(w))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)#生成随机噪音

    return X,y.reshape(-1,1)#将y作为列向量返回
    
    

true_w=torch.tensor([2,-3.4])
true_b=4.2

features,labels=synthic_data(true_w,true_b,1000)

def data_iter(batch_size, features, labels):
    # 获取特征数据的数量
    nums = len(features)
    # 创建一个从0到nums-1的索引列表
    idx = list(range(nums))
    # 打乱索引列表的顺序以实现数据的随机性
    random.shuffle(idx)
    # 遍历从0到nums（不包括nums），每次步进batch_size
    for i in range(0, nums, batch_size):
        # 批处理索引生成，确保索引不会超出数据范围
        batch_idx=idx[i:min(i+batch_size,nums)]
    yield features[batch_idx], labels[batch_idx] #只有tensor才能用切片访问

batch_size=10

#初始化模型参数
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

#定义线性回归模型,x:1000*2,w:2*1
def linreg(X,w,b):
    return torch.mm(X,w)+b

#损失函数使用平方误差
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

# 优化算法-梯度下降-随机梯度下降

def sgd(params,lr,batch_size):
    #更新时不要进行梯度计算
    with torch.no_grad():
        for p in params:
            # 根据学习率和梯度更新参数 p
            p-=lr*p.grad/batch_size
            p.grad.zero_()#手动把梯度设置为0


epochs=3
lr=0.03

for epoch in range(epochs):
    for x,y in data_iter(batch_size,features,labels):
        l=squared_loss(linreg(x,w,b),y)
        # 计算张量 l 的总和并对总和进行反向传播
        l.sum().backward()
        sgd([w,b],lr,batch_size)
     
    with torch.no_grad():
        train_l=squared_loss(linreg(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
# 此代码实现了一个简单的线性回归模型训练过程
# 遍历指定的训练轮数，每轮中通过数据迭代器获取批量数据
# 计算预测值与真实标签的平方损失，并进行反向传播
# 使用随机梯度下降法更新模型参数
# 在每轮训练结束后，计算并打印训练集上的平均损失值
