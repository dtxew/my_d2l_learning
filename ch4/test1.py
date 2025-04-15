import torch
#softmax虽然是回归，本质是分类问题
#分类与回归的区别：输出是可能有多个离散值，每个值是置信度

#均方损失
#可对类别进行1位有效编码
#通常使用交叉熵损失

import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans=transforms.ToTensor()#把图片转为tensor,transforms=trans

#训练数据集
mnist_tran=torchvision.datasets.FashionMNIST("ch4/data",train=True
                                             ,transform=trans,download=True)
#测试数据集
mnist_test=torchvision.datasets.FashionMNIST("ch4/data",train=False,transform=trans,download=True)

#区别：测试数据集是用来测试训练模型好坏的

