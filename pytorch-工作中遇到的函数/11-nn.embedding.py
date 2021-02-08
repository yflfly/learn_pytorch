# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

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

'''
可见，这个list的元素成了一个个list。还要做一步：将上面的三个list按单词数从多到少排列。标点也算单词。至于为什么，后面会说到。
那就变成了：
'''
batch = [['i', 'am', 'a', 'boy', '.'], ['i', 'am', 'very', 'lucky', '.'], ['how', 'are', 'you', '?']]

# 可见，每个句子的长度，即每个内层list的元素数为：5,5,4。这个长度也要记录。
lens = [5, 5, 4]

'''
之后，为了能够处理，将batch的单词表示转为在词典中的index序号，这就是word2id的作用。
转换过程很简单，假设转换之后的结果如下所示，当然这些序号是我编的。
'''
batch = [[3, 6, 5, 6, 7], [6, 4, 7, 9, 5], [4, 5, 8, 7]]

# 同时，每个句子结尾要加EOS，假设EOS在词典中的index是1。
batch = [[3, 6, 5, 6, 7, 1], [6, 4, 7, 9, 5, 1], [4, 5, 8, 7, 1]]

# 那么长度要更新：
lens = [6, 6, 5]
'''
很显然，这个mini-batch中的句子长度不一致！所以为了规整的处理，对长度不足的句子，进行填充。\
填充PAD假设序号是2，填充之后为：
'''
batch = [[3, 6, 5, 6, 7, 1], [6, 4, 7, 9, 5, 1], [4, 5, 8, 7, 1, 2]]
'''
这样就可以直接取词向量训练了吗?
不能！上面batch有3个样例，RNN的每一步要输入每个样例的一个单词，一次输入batch_size个样例，
所以batch要按list外层是时间步数(即序列长度)，list内层是batch_size排列。即batch的维度应该是：
[seq_len,batch_size]
[seq_len,batch_size]
[seq_len,batch_size]
'''

'''
怎么变换呢？变换方法可以是：使用itertools模块的zip_longest函数。
而且，使用这个函数，连填充这一步都可以省略，因为这个函数可以实现填充！
'''
PAD = 2
batch = list(itertools.zip_longest(batch, fillvalue=PAD))
# fillvalue就是要填充的值，强制转成list
# 经过变换，结果应该是：
batch = [[3, 6, 4], [6, 4, 5], [5, 7, 8], [6, 9, 7], [7, 5, 1], [1, 1, 2]]

# batch还要转成LongTensor：
batch = torch.LongTensor(batch)

'''
这里的batch就是词向量层的输入。
词向量层的输出是什么样的？
好了，现在使用建立了的embedding直接通过batch取词向量了，如：
'''
embed_batch = embed(batch)

'''
维度的前两维和前面讲的是一致的。可见多了一个第三维，这就是词向量维度。所以，Embedding层的输出是：
[seq_len,batch_size,embedding_size]
'''
