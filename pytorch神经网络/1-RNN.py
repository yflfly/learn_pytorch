# coding:utf-8
import torch
import torch.nn as nn

# 构造RNN网络，x的维度5，隐层的维度10,网络的层数2
rnn_seq = nn.RNN(5, 10, 2)
# 构造一个输入序列，句长为 6，batch 是 3， 每个单词使用长度是5的向量表示
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
对应参数的解释：
eq_len是一个句子的最大长度，比如6
input_dim是输入的维度，比如是5
batch_size是一次往RNN输入句子的数目，比如是3

注意：
RNN输入的是序列，一次把批次的所有句子都输入了，得到的ouptut和hidden都是这个批次的所有的输出和隐藏状态，维度也是三维。

可以理解为现在一共有batch_size个独立的RNN组件，RNN的输入维度是input_dim，总共输入seq_len个时间步，
则每个时间步输入到这个整个RNN模块的维度是[batch_size,input_dim]

'''
