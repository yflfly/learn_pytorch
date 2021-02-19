# coding:utf-8
import numpy
import torch

'''
tensor 和 numpy 的互相转换
为什么要相互转换：
简单一句话, numpy操作多样, 简单. 但网络前向只能是tensor类型, 各有优势, 所以需要相互转换补充.
'''
# convert Tensor x of torch to array y of  numpy:
y = x.numpy()

# convert array x of  numpy to Tensor y of torch:
y = torch.from_numpy(x)

# 先将数据转换成Tensor, 再使用CUDA函数来将Tensor移动到GPU上加速

# 如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
x_np = x.data.numpy()

# 改为：

x_np = x.data.cpu().numpy()