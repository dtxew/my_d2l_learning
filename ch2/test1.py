import torch


a=torch.arange(20,dtype=torch.float32).reshape(5,4)
print(a)

#转置
A=a.T

print(A)

#求平均值
print(A.mean())

#按维度求和,将每个特定维度加起来组成新的数组
b=torch.arange(20).reshape(4,5)
print(b)
print(b.sum(axis=0))
print(b.sum(axis=1))
print(b.sum(axis=1,keepdim=True))#不丢失维度

#求数组点乘torch.dot(x,y)
#线代中的矩阵乘法torch.mm(x,y)
#矩阵和向量乘法torch.mv(M,V)
