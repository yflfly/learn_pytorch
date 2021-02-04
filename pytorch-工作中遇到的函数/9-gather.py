# coding:utf-8
import torch

'''
内容来源网址;
https://blog.csdn.net/leviopku/article/details/108735704
gather，顾名思义，聚集，集合，有点像军训的时候，排队一样，把队伍按照教官想要的顺序进行排列
还有一个更恰当的比喻：gather的作用是根据索引查找，然后将查找的结果以张量矩阵的形式返回
'''