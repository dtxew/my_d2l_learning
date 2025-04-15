import torch

# 自动求导用于计算一个函数在指定值上的导数
# 实现：无环有向图,链式法则，正向和反向累积

x=torch.arange(4.0,requires_grad=True)#第二个参考告诉x存储梯度
#f(x)=2xTx

y=2*torch.dot(x,x)

print(y)

#调用反向传播计算y关于每一个分量x梯度

y.backward()
print(x.grad)

#torch会累积梯度，每次计算前都要清空

# x.grad.zero_()
y=torch.sum(x)

y.backward()
print(x.grad)