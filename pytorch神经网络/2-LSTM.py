# coding:utf-8
import torch
import torch.nn as nn

# 输入维度 50，隐层100维，两层
lstm_seq = nn.LSTM(50, 100, num_layers=2)
# 输入序列seq= 10，batch =3，输入维度=50
lstm_input = torch.randn(10, 3, 50)

out, (h, c) = lstm_seq(lstm_input)  # 使用默认的全 0 隐藏状态
print('out', out.shape)  # torch.Size([10, 3, 100])
print('h', h.shape)  # torch.Size([2, 3, 100])  参考[num_layers * num_directions, batch, hidden_size]
print('c', c.size())  # torch.Size([2, 3, 100])

'''
问题1：out和(h,c)的size各是多少？
回答：out：(10 * 3 * 100)，(h,c)：都是(2 * 3 * 100)
问题2：out[-1,:,:]和h[-1,:,:]相等吗？
回答： 相等
'''