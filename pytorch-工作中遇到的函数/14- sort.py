# coding:utf-8
import torch
import torch.nn as nn

'''
方法原型：
torch.sort(input, dim=None, descending=False, out=None) -&gt; (Tensor, LongTensor)
'''
'''
返回值：
A tuple of (sorted_tensor, sorted_indices) is returned, 
where the sorted_indices are the indices of the elements in the original input tensor.
'''

'''
参数：
input (Tensor) – the input tensor 形式上与 numpy.narray 类似
dim (int, optional) – the dimension to sort along 维度，对于二维数据：dim=0 按列排序，dim=1 按行排序，默认 dim=1
descending (bool, optional) – controls the sorting order (ascending or descending)
    降序，descending=True 从大到小排序，descending=False 从小到大排序，默认 descending=Flase
'''