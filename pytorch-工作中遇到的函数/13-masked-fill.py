# coding:utf-8
import torch

attn = torch.randn(3, 3)
print(attn)

mask = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(mask)

# 将mask矩阵中为0的值填充为-1e9
attn = attn.masked_fill(mask == 0, -1e9)
print(attn)

'''
输出结果如下所示：
tensor([[-0.2498, -0.0787,  0.5016],
        [-0.5436, -1.1036, -0.4536],
        [-0.4595,  1.2925, -1.3853]])
        
tensor([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        
tensor([[-2.4981e-01, -1.0000e+09, -1.0000e+09],
        [-1.0000e+09, -1.1036e+00, -1.0000e+09],
        [-1.0000e+09, -1.0000e+09, -1.3853e+00]])
'''
