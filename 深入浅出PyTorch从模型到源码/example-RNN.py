# coding:utf-8
import torch
import torch.nn as nn

rnn = nn.RNN(10, 20, 2)  # 10：词向量的维度  20：隐含变量的维度大小  2：循环神经网络的层数
input = torch.randn(5, 3, 10)  # 序列对应的张量，序列的长度*迷你批次的大小*输入的特征数目
h0 = torch.randn(2, 3, 20)  # 初始的隐状态 2：循环神经网络的层数 3：迷你批次大小  20：隐含层的维度
output, hn = rnn(input, h0)

print(rnn)
print(input.shape)
print(h0.shape)
print(output.shape)
print(hn.shape)
'''
课本P276
输出结果如下所示：
RNN(10, 20, num_layers=2)
torch.Size([5, 3, 10])
torch.Size([2, 3, 20])
torch.Size([5, 3, 20])
torch.Size([2, 3, 20])
'''
