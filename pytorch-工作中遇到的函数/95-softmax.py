# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
输出的是一个概率分布：每一个元素都非负且和为1
你也可以认为这只是对输入元素进行的求幂运算符，使所有的内容都非负，然后除以规范化常量
'''
data = torch.randn(5)
print(data)  # tensor([ 0.7319, -0.3785, -0.2287,  0.3616,  1.5175])
print(F.softmax(data, dim=0))  # tensor([0.2176, 0.0717, 0.0833, 0.1502, 0.4773])
print(F.softmax(data, dim=0).sum())  # tensor(1.) 总和为1，因为它是一个分布
print(F.log_softmax(data, dim=0))  # tensor([-1.5252, -2.6357, -2.4858, -1.8956, -0.7397])

