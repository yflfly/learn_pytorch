# coding:utf-8
import torch
import torch.nn as nn

# 构造RNN网络，x的维度5，隐层的维度10,网络的层数2
rnn_seq = nn.RNN(5, 10, 2)
# 构造一个输入序列，句长为 6，batch 是 3， 每个单词使用长度是 5的向量表示
x = torch.randn(6, 3, 5)
# out,ht = rnn_seq(x,h0)
out, ht = rnn_seq(x)  # h0可以指定或者不指定

print('out', out.shape)  # torch.Size([6, 3, 10])
print('ht', ht.shape)  # torch.Size([2, 3, 10])

'''
对于最简单的RNN，我们可以使用两种方式来调用,torch.nn.RNNCell(),它只接受序列中的单步输入，必须显式的传入隐藏状态。
torch.nn.RNN()可以接受一个序列的输入，默认会传入一个全0的隐藏状态，也可以自己申明隐藏状态传入。
'''

'''
输入大小是三维tensor[seq_len,batch_size,input_dim]
'''
