# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Module类是nn模块里提供的一个模型构造类，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型。
下面继承Module类构造本节开头提到的多层感知机。
这里定义的MLP类重载了Module类的__init__函数和forward函数。
它们分别用于创建模型参数和定义前向计算。前向计算也即正向传播。
'''
import torch
from torch import nn


class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


X = torch.rand(2, 784)
print(X)
'''
tensor([[0.1330, 0.2617, 0.3695,  ..., 0.9918, 0.4108, 0.2036],
        [0.1520, 0.2079, 0.3169,  ..., 0.1419, 0.6782, 0.3243]])
'''
print('*' * 100)
net = MLP()
print(net)

'''
MLP(
  (hidden): Linear(in_features=784, out_features=256, bias=True)
  (act): ReLU()
  (output): Linear(in_features=256, out_features=10, bias=True)
)
'''
print(net(X))
'''
tensor([[ 0.2447, -0.1285, -0.1711,  0.0660,  0.0790, -0.3839, -0.0752, -0.0051,
         -0.0923,  0.0612],
        [ 0.3492, -0.1122, -0.1035, -0.0063,  0.1694, -0.3175, -0.0643,  0.0057,
          0.0328,  0.0272]], grad_fn=<AddmmBackward>)
'''
