# coding:utf-8
from __future__ import print_function
import torch

# 张量
# Tensor（张量）类似于NumPy的ndarray，但还可以在GPU上使用来加速计算
# 1、创建一个没有初始化的5*3矩阵：
x = torch.empty(5, 3)
print(x)
# 输出结果
'''
tensor([[2.2391e-19, 4.5869e-41, 1.4191e-17],
        [4.5869e-41, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])
'''
# 2、创建一个随机初始化矩阵：
x = torch.rand(5, 3)
print(x)

# 输出结果
'''
tensor([[0.5307, 0.9752, 0.5376],
        [0.2789, 0.7219, 0.1254],
        [0.6700, 0.6100, 0.3484],
        [0.0922, 0.0779, 0.2446],
        [0.2967, 0.9481, 0.1311]])
'''
# 构造一个填满0且数据类型为long的矩阵:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 输出结果
'''
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
'''
# 直接从数据构造张量：
x = torch.tensor([5.5, 3])
print(x)

# 输出结果
'''
tensor([5.5000, 3.0000])
'''
# 根据已有的tensor建立新的tensor。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如dtype等：
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 重载 dtype
print(x)                                      # 结果size一致

# 输出结果
'''
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 1.6040, -0.6769,  0.0555],
        [ 0.6273,  0.7683, -0.2838],
        [-0.7159, -0.5566, -0.2020],
        [ 0.6266,  0.3566,  1.4497],
        [-0.8092, -0.6741,  0.0406]])
'''
# 获取张量的形状：
print(x.size())
# torch.Size([5, 3])