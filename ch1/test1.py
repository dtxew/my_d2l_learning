import torch
import numpy as np

# 张量
x=torch.arange(12)

print(x)
print(x.shape)# 访问x的形状
X=x.reshape(3,4)#改变形状
print(X)

#zeros,ones创建全0或全1数组


#利用列表创建
x=torch.tensor([[1,2],[3,4]])
print(x)

#可以进行+-*/**等运算

# 可以将两个数组进行拼接
x=torch.arange(12,dtype=torch.float32).reshape(3,4)
y=torch.tensor([[1.1,2,3,4],[5,6,7,8],[9,9,9,9]])

print(torch.cat((x,y),dim=0))#堆起来
print(torch.cat((x,y),dim=1))#横着放

# 可以将整个数组相加
print(x.sum())

#广播机制
a=torch.arange(2).reshape(1,2)
b=torch.arange(3).reshape(3,1)

print(a+b)

#可以转换为numpy
A=a.numpy()
