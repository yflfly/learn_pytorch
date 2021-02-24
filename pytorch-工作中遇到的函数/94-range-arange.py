# coding:utf-8
import torch

'''
torch.range(start=1, end=6) 的结果是会包含end的，
而torch.arange(start=1, end=6)的结果并不包含end。
两者创建的tensor的类型也不一样。
'''
x = torch.range(1, 6)
print(x)
print(x.dtype)
print('-----------------分隔符-----------------')
y = torch.arange(1, 6)
print(y)
print(y.dtype)
'''
输出结果如下所示：
tensor([1., 2., 3., 4., 5., 6.])
torch.float32
-----------------分隔符-----------------
tensor([1, 2, 3, 4, 5])
torch.int64
'''