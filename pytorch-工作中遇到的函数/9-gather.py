# coding:utf-8
import torch

'''
内容来源网址;
https://blog.csdn.net/leviopku/article/details/108735704
gather，顾名思义，聚集，集合，有点像军训的时候，排队一样，把队伍按照教官想要的顺序进行排列
还有一个更恰当的比喻：gather的作用是根据索引查找，然后将查找的结果以张量矩阵的形式返回

参考网址：https://zhuanlan.zhihu.com/p/101896024
'''
# 创建input的张量，是output的数据来源
a = torch.randint(1, 50, size=(2, 3))
print(a)

# 创建一个index,最后咱们的输出张量的维度一定是和index的维度是相同的
index = torch.LongTensor([[0, 1, 0], [1, 0, 1]])

b = torch.gather(a, 1, index)
print(b)

c = torch.gather(a, 0, index)

print(c)

'''
输出结果如下所示：
a:
tensor([[41, 35,  4],
        [32, 45, 39]])
b:
tensor([[41, 35, 41],
        [45, 32, 45]])
c:
tensor([[41, 45,  4],
        [32, 35, 39]])
'''