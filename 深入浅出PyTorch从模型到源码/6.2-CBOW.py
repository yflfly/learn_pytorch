# coding:utf-8
import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear1 = nn.Linear(context_size * embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        embedded = embedded.view(embedded.size(0), -1)
        hid = torch.relu(self.linear1(embedded))
        out = self.linear2(hid)
        return out


# 定义CBOW的训练代码
def train_cbow():
    hidden_size = 128
    vocab_size = 10000
    embed_size = 100
    context_size = 4
    learning_rate = 0.02
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    model = CBOW(vocab_size, embed_size, context_size, hidden_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(n_epoch):
        for context, target in cbow_train:
            model.zero_grad()
            logits = model(context)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
    return model


'''
context_size：定义了上下文单词的数目，比如，中心单词周围有4个单词，则这个值为4
如果需要计算单词表中某一个单词的概率，可以使用Softmax对线性变换的张量进行计算，如果需要计算损失函数，可以使用交叉熵计算输出某个特定单词的损失函数
'''
'''
在代码中使用了CrossEntropyLoss，根据具体的上下文输出结果logits和预测目标target计算对应的损失函数，最后进行反向传播，优化对应模型的参数，其中包括模型中的embedding属性，对应的是基于连续词袋的word2vec模型的训练结果
'''