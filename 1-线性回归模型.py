# coding:utf-8
import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim

        self.weight = nn.Parameter(torch.randn(ndim, 1))  # 定义权重
        self.bias = nn.Parameter(torch.randn(1))  # 定义偏置

    def forward(self, x):
        return x.mm(self.weight) + self.bias


lm = LinearModel(5)  # 定义线性回归模型，特征数为5
x = torch.randn(4, 5)  # 定义随机输入，迷你批次大小为4
print(lm(x))  # 得到每个迷你批次的输出

'''
输出结果如下所示：
tensor([[ 0.0997],
        [ 1.9427],
        [ 1.6200],
        [-0.3356]], grad_fn=<AddBackward0>)
'''
