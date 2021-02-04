# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
为什么使用pack_padded_sequence
在使用深度学习特别是LSTM进行文本分析时，经常会遇到文本长度不一样的情况，
此时就需要对同一个batch中的不同文本使用padding的方式进行文本长度对齐，
方便将训练数据输入到LSTM模型进行训练，同时为了保证模型训练的精度，
应该同时告诉LSTM相关的padding情况，此时pytorch中的pack_padded_sequence就有了用武之地
'''

# toydata 数据如下：batch_size=3，max_seq_len=7，padding用0表示。
sample = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 0, 0, 0], [1, 2, 3, 4, 5, 0, 0]]).T
print(sample)
# 存储句子的长度
lengths = (7, 4, 5)

# pack_padded_sequence
'''
这个函数用通俗的话说是将pad好了的sequence解开，恢复为原来没有pad的样子
pack_padded_sequence(inputs, lengths, batch_first=False, enforce_sorted=True)
输入：
1）inputs (Tensor): pad好了的sequence集，数据的size要求为[T, B, *]，
其中T表示的是所有句子中最长的句子的长度，B为Batch的大小，要求inputs至少是两维；
2）lengths (Tensor or list or tuple or …): 储存了每个句子没有pad之前的长度的集合
3）batch_first: 如果是True的话，就把B放在T前
4）enforce_sorted: 如果是True的话就说明输入的inputs已经按照句子长度递减排好了，False的话就要在函数里排。
'''
pack_sample = pack_padded_sequence(sample, lengths, enforce_sorted=False)

print(f"pack_sample.data = {pack_sample.data}")
print(f"pack_sample.batch_sizes = {pack_sample.batch_sizes}")
print(f"pack_sample.sorted_indices = {pack_sample.sorted_indices}")
print(f"pack_sample.unsorted_indices = {pack_sample.unsorted_indices}")

'''
结果输出如下所示：
pack_sample.data = tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7])
pack_sample.batch_sizes = tensor([3, 3, 3, 3, 2, 1, 1])
pack_sample.sorted_indices = tensor([0, 2, 1])
pack_sample.unsorted_indices = tensor([0, 2, 1])
'''

print('-' * 100)
# pad_packed_sequence 这个函数就是上一个函数的逆过程。
pad_pack_sample, lengths = pad_packed_sequence(pack_sample)
print(pad_pack_sample)
'''
tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 0, 5],
        [6, 0, 0],
        [7, 0, 0]])
'''
