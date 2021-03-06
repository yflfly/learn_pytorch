# coding:utf-8
import torch

tensor = torch.randn(3, 4, 5)
print(tensor.type())  # 数据类型
print(tensor.size())  # 张量的shape，是个元组
print(tensor.dim())  # 维度的数量
print(tensor)
'''
输出的结果如下所示：
torch.FloatTensor

torch.Size([3, 4, 5])

3

tensor([[[-1.1305, -0.7729,  1.6582, -0.3064,  0.1728],
         [-1.3618, -1.0769, -0.6611,  1.3109, -0.6923],
         [ 1.8063,  0.3214,  0.1584, -0.0227,  0.4814],
         [ 0.5433,  0.9705, -1.2959,  2.0936,  0.6737]],

        [[ 0.5990,  1.6877,  0.0989,  0.5488, -0.1184],
         [ 0.8136,  0.6946,  0.0092, -0.6920, -0.2138],
         [ 0.2646, -0.6358,  1.3143, -1.3735,  3.3871],
         [-1.2083, -0.9601, -0.1624,  1.1612, -1.4391]],

        [[ 0.0803, -0.5388, -0.3639, -1.0977, -1.7307],
         [ 1.1189,  0.4182,  0.6489, -0.8255, -1.0510],
         [-0.0232,  1.3597,  0.2838,  0.3775, -0.8842],
         [-1.5845, -0.2684,  0.0385,  0.0283,  0.6274]]])
'''
