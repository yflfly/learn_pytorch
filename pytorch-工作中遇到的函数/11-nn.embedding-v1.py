# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# 在pytorch里面实现word embedding是通过一个函数来实现的:nn.Embedding


word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 5)  # 2*5的矩阵  2：表示词表的长度  5：表示生成向量的维度
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)

'''
这就是我们输出的hello这个词的word embedding，代码会输出如下内容
tensor([[ 0.9561,  0.3167,  0.6582,  0.4647, -1.2379]],
       grad_fn=<EmbeddingBackward>)
'''

'''
解释说明：
首先我们需要word_to_ix = {'hello': 0, 'world': 1}，每个单词我们需要用一个数字去表示他，
这样我们需要hello的时候，就用0来表示它。

接着就是word embedding的定义nn.Embedding(2, 5)，这里的2表示有2个词，5表示5维度，其实也就是一个2x5的矩阵，
所以如果你有1000个词，每个词希望是100维，你就可以这样建立一个word embedding，nn.Embedding(1000, 100)。
如何访问每一个词的词向量是下面两行的代码，注意这里的词向量的建立只是初始的词向量，并没有经过任何修改优化，
我们需要建立神经网络通过learning的办法修改word embedding里面的参数使得word embedding每一个词向量能够表示每一个不同的词
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx = Variable(hello_idx)
'''

'''
接着这两行代码表示得到一个Variable，它的值是hello这个词的index，也就是0。
这里要特别注意一下我们需要Variable，因为我们需要访问nn.Embedding里面定义的元素，
并且word embeding算是神经网络里面的参数，所以我们需要定义Variable。

hello_embed = embeds(hello_idx)这一行表示得到word embedding里面关于hello这个词的初始词向量，
最后我们就可以print出来。
'''