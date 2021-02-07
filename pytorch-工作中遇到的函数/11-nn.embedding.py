# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
函数调用形式：
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
 max_norm=None,  norm_type=2.0,   scale_grad_by_freq=False, 
 sparse=False,  _weight=None)

解释：
其为一个简单的存储固定大小的词典的嵌入向量的查找表，意思就是说，给定一个编号，嵌入层就能返回这个编号对应的嵌入向量，
嵌入向量反映了各个编号代表的符号之间的语义关系
输入为一个编号列表，输出对应的符号嵌入向量

参数说明：
num_embeddings (python:int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
embedding_dim (python:int) – 嵌入向量的维度，即用多少维来表示一个符号。
padding_idx (python:int, optional) – 填充id，比如，输入长度为100，但是每次的句子长度并不一样，
                后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，
                就不会计算其与其它符号的相关性。（初始化为0）
max_norm (python:float, optional) – 最大范数，如果嵌入向量的范数超过了这个界限，就要进行再归一化。
norm_type (python:float, optional) – 指定利用什么范数计算，并用于对比max_norm，默认为2范数。
scale_grad_by_freq (boolean, optional) – 根据单词在mini-batch中出现的频率，对梯度进行放缩。默认为False.
sparse (bool, optional) – 若为True,则与权重矩阵相关的梯度转变为稀疏张量。


链接：https://www.jianshu.com/p/63e7acc5e890

'''
# 下面是关于Embedding的使用
'''
torch.nn包下的Embedding，作为训练的一层，随模型训练得到适合的词向量
'''
# 建立词向量层
n_vocabulary = 100
embedding_size = 5
embed = torch.nn.Embedding(n_vocabulary, embedding_size)
print(embed)  # Embedding(100, 5)

'''
找到对应的词向量放进网络：词向量的输入应该是什么样子

实际上，上面通过随机初始化建立了词向量层后，建立了一个“二维表”，存储了词典中每个词的词向量。
每个mini-batch的训练，都要从词向量表找到mini-batch对应的单词的词向量作为RNN的输入放进网络。
那么怎么把mini-batch中的每个句子的所有单词的词向量找出来放进网络呢，输入是什么样子，输出是什么样子？
'''

'''
首先我们知道肯定先要建立一个词典，建立词典的时候都会建立一个dict：word2id：存储单词到词典序号的映射。
假设一个mini-batch如下所示：

'''
lst = ['I am a boy.', 'How are you?', 'I am very lucky.']  # 这个mini-batch有3个句子，即batch_size=3

# 第一步首先要做的是：将句子标准化，所谓标准化，指的是：大写转小写，标点分离，这部分很简单就略过。经处理后，mini-batch变为：
lst_1 = [['i', 'am', 'a', 'boy', '.'], ['how', 'are', 'you', '?'], ['i', 'am', 'very', 'lucky', '.']]




