# coding:utf-8
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 创建一个Dataframe
    data = pd.DataFrame(np.arange(16).reshape(4, 4), index=list('abcd'), columns=list('ABCD'))
    print('data', data)
    # 取索引为'a'的行
    print('data.loc', data.loc['a'])
    # 取第一行数据，索引为'a'的行就是第一行，所以结果相同
    print('data.iloc', data.iloc[0])
'''
data     
    A   B   C   D
a   0   1   2   3
b   4   5   6   7
c   8   9  10  11
d  12  13  14  15

data.loc 
A    0
B    1
C    2
D    3
Name: a, dtype: int32

data.iloc 
A    0
B    1
C    2
D    3
Name: a, dtype: int32

'''
