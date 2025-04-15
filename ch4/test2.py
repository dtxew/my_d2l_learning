import torch
from IPython import display
from d2l import torch as d2l
from torchvision import transforms
from torch.utils import data
import torchvision

batch_size=256

trans=transforms.ToTensor()

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


#将它们拉成28*28=784的向量（长宽都为28）,输入为784
num_inputs=784
num_outputs=10
#有10个分类，输出为10

#初始化权重和偏置
W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)

def softmax(X):
    X_exp=torch.exp(X)
    #将第二个维度设置为1求和
    #比如数据为(784,10),求和后就是(784,1)，即是按行求和
    xsum=X_exp.sum(1,keepdim=True)#keepdim将保留维度
    
    return X_exp/xsum #利用广播机制

#定义神经网络，可以参考电子书
def net(X):
    return softmax(torch.mm(X.reshape(-1,W.shape[0]),W)+b)

# #设真实值对应下标关系y=[0,1]y[0]=0表示0号元素正确分类是0
# #设预测概率为y_hat=[[0.1,0.8,0.1][0.1,0.7,0.2]],y[0][0]表示0号元素对标签0的预测概率为0.1
# #问题来了，怎么拿出来呢？

# y=torch.tensor([0,2])
# y1=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])

# print(y1[[0,1],y])
# print(y1[[0,1],[0,2]])
# #拿出00,12这两个元素

y=torch.tensor([0,2])
y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])

#交叉熵损失函数,书p64
#y_hat是预测概率，y是真实标签
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

#计算分类准确率，参考P72



def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and (y_hat.shape[1])>1:
        y_hat=y_hat.argmax(axis=1)
        cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())


"""
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
"""
#评估模型net的准确率
def evaluate_accuracy(net, data_iter):
    metric=d2l.Accumulator(2)
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())#numel返回张量中元素总数
    return metric[0]/metric[1]#准确率除以元素总数


def train_epoch(net,train_iter,loss,updater):
    metric=d2l.Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        
        l.sum().backward()
        updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]


lr=0.1

def updater(lr):
    return d2l.sgd([W,b],lr,batch_size)

def train_main(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics=train_epoch(net,train_iter,loss,updater)
        acc=evaluate_accuracy(net,test_iter)
        train_loss,train_acc=train_metrics
        print(f'Epoch {epoch + 1}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test acc: {acc:.4f}')

if __name__=='__main__':
    train_main(net,train_iter,test_iter,cross_entropy,3,updater)