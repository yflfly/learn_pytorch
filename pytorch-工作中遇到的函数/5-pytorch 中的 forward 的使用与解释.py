# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
pytorch 中的 forward 的使用与解释
最近在使用pytorch的时候，模型训练时，不需要使用forward，
只要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数
'''


# forward 使用
class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # ......

    def forward(self, x):
        # ......
        return x


data = '训练神经网络'  # 输入数据
# 实例化一个对象
module = Module()
# 前向传播
print(module(data))
# 而不是使用下面的
# module.forward(data)
'''
实际上 module(data) 等价于 module.forward(data)
'''