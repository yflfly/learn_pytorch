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

# forward函数的解释
'''
等价的原因是因为 python calss 中的__call__和__init__方法.
'''


class A_1():


    def __call__(self):
        print('i can be called like a function')


a = A_1()
a()  # 输出结果i can be called like a function


# __call__里调用其他的函数


class A():
    def __call__(self, param):
        print('i can called like a function')
        print('传入参数的类型是：{}   值为： {}'.format(type(param), param))

        res = self.forward(param)
        return res

    def forward(self, input_):
        print('forward 函数被调用了')

        print('in  forward, 传入参数类型是：{}  值为: {}'.format(type(input_), input_))
        return input_


a = A()

input_param = a('i')
print("对象a传入的参数是：", input_param)

'''
输出的结果：
i can called like a function
传入参数的类型是：<class 'str'>   值为： i
forward 函数被调用了
in  forward, 传入参数类型是：<class 'str'>  值为: i
对象a传入的参数是： i
'''