# coding:utf-8
import torch

# 在将卷积层输入全连接层的情况下通常需要对张量做形变处理，
# 相比torch.view，torch.reshape可以自动处理输入张量不连续的情况。
tensor = torch.rand(2, 3, 4)
print(tensor.shape)
shape = (6, 4)
tensor = torch.reshape(tensor, shape)
print(tensor.shape)

'''
输出结果如下所示：
torch.Size([2, 3, 4])
torch.Size([6, 4])
'''