# coding:utf-8
import torch

'''
先看torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。
squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。
a.squeeze(N) 就是去掉a中指定的维数为一的维度。
还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。

再看torch.unsqueeze()这个函数主要是对数据维度进行扩充。
给指定位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。
a.unsqueeze(N) 就是在a中指定位置N加上一个维数为1的维度。
还有一种形式就是b=torch.unsqueeze(a，N) a就是在a中指定位置N加上一个维数为1的维度
'''

a = torch.randn(1, 3)

print(a)  # tensor([[-0.5883,  0.7447, -0.3127]]) 每次运行输出的结果可能会有不同
print(a.shape)  # torch.Size([1, 3])

b = torch.unsqueeze(a, 1)
print(b)
print(b.shape)  # torch.Size([1, 1, 3])

c = a.unsqueeze(0)
print(b)
print(c.shape)  # torch.Size([1, 1, 3])

print('-------------------------------')
d = torch.squeeze(c)
print(d)
print(d.shape)  # torch.Size([3])

'''
输出的结果如下所示：
tensor([[ 0.8719, -1.0836,  0.4253]])
torch.Size([1, 3])
tensor([[[ 0.8719, -1.0836,  0.4253]]])
torch.Size([1, 1, 3])
tensor([[[ 0.8719, -1.0836,  0.4253]]])
torch.Size([1, 1, 3])
-------------------------------
tensor([ 0.8719, -1.0836,  0.4253])
torch.Size([3])
'''
