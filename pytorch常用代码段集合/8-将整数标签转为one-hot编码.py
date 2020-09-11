# coding:utf-8
import torch

# pytorch的标记默认从0开始
tensor = torch.tensor([0, 2, 1, 3])
print(tensor)  # tensor([0, 2, 1, 3])
N = tensor.size(0)
print(N)  # 4
num_classes = 4
one_hot = torch.zeros(N, num_classes).long()
print(one_hot)
print(one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long()))
'''
输出的结果如下所示：
tensor([0, 2, 1, 3])
4
tensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])
tensor([[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])

'''
