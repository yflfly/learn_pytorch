# coding:utf-8
import torch

'''
官方文档：
torch.transpose(input, dim0, dim1, out=None) → Tensor

函数返回输入矩阵input的转置。交换维度dim0和dim1
参数:
input (Tensor) – 输入张量，必填
dim0 (int) – 转置的第一维，默认0，可选
dim1 (int) – 转置的第二维，默认1，可选
'''
# 创造二维数据x，dim=0时候2，dim=1时候3
x = torch.randn(2, 3)  # 'x.shape  →  [2,3]'
# 创造三维数据y，dim=0时候2，dim=1时候3，dim=2时候4
y = torch.randn(2, 3, 4)  # 'y.shape  →  [2,3,4]'

print(x.size())  # ([2, 3])
print(y.size())  # ([2, 3, 4])
print('------------------')
# 对于transpose
z1 = x.transpose(0, 1)  # 'shape→[3,2] '
print(x.size())  # ([2, 3])
print('z1', z1.size())  # [3, 2])

x.transpose(1, 0)  # 'shape→[3,2] '
print(x.size())  # ([2, 3])

y1 = y.transpose(0, 1)  # 'shape→[3,2,4]'
print(y.size())  # ([2, 3, 4])
print('y1', y1.size())  # ([3, 2, 4])

'''
输出结果如下所示：
torch.Size([2, 3])
torch.Size([2, 3, 4])
------------------
torch.Size([2, 3])
z1 torch.Size([3, 2])
torch.Size([2, 3])
torch.Size([2, 3, 4])
y1 torch.Size([3, 2, 4])

记住：转置之后的tensor进行赋值给新的变量
'''
