# coding:utf-8
import torch
import torch.nn as nn

# GRU比较像传统的RNN

gru_seq = nn.GRU(10, 20, 2)  # x_dim,h_dim,layer_num
gru_input = torch.randn(3, 32, 10)  # seq，batch，x_dim
out, h = gru_seq(gru_input)

print('out', out.shape)  # torch.Size([3, 32, 20]) 参考 [num_layers * num_directions, batch, hidden_size]
print('h', h.shape)  # torch.Size([2, 32, 20])
