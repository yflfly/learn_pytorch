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

# 例子
x = torch.randn(3, 4)
print(x)  # 初始值，始终不变

sorted, indices = torch.sort(x)  # 按行从小到大排序
print(sorted)
print(indices)

'''
输出结果如下所示：
tensor([[ 0.8529, -0.2604,  0.5451, -0.3952],
        [-0.0424, -0.1874,  0.1758,  0.1154],
        [ 1.3341,  0.2611, -0.3443, -0.3940]])
tensor([[-0.3952, -0.2604,  0.5451,  0.8529],
        [-0.1874, -0.0424,  0.1154,  0.1758],
        [-0.3940, -0.3443,  0.2611,  1.3341]])
tensor([[3, 1, 2, 0],
        [1, 0, 3, 2],
        [3, 2, 1, 0]])
'''