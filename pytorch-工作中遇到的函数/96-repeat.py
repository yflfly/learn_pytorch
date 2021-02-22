# coding:utf-8
import torch
import torch.nn.functional as F
import numpy as np

'''
repeat 相当于一个broadcasting的机制

repeat(*sizes)

沿着指定的维度重复tensor
'''
a = torch.Tensor(2, 1, 4)
B = a.repeat(1, 3, 1)
print(a.shape)  # torch.Size([2, 1, 4])
print(B.shape)  # torch.Size([2, 3, 4])

print(a)
print('-----------------------------')
print(B)
'''
tensor([[[ 5.8277e-05, -6.3343e+16,  6.1630e-33,  7.0362e+22]],

        [[ 7.5632e+28,  6.7340e+22,  6.7120e+22,  2.8120e+29]]])
-----------------------------
tensor([[[ 5.8277e-05, -6.3343e+16,  6.1630e-33,  7.0362e+22],
         [ 5.8277e-05, -6.3343e+16,  6.1630e-33,  7.0362e+22],
         [ 5.8277e-05, -6.3343e+16,  6.1630e-33,  7.0362e+22]],

        [[ 7.5632e+28,  6.7340e+22,  6.7120e+22,  2.8120e+29],
         [ 7.5632e+28,  6.7340e+22,  6.7120e+22,  2.8120e+29],
         [ 7.5632e+28,  6.7340e+22,  6.7120e+22,  2.8120e+29]]])
'''
