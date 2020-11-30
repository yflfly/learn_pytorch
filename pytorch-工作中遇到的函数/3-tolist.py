# coding:utf-8
import torch
'''
tolist:
作用：将tensor转换为list数据
'''


if __name__ == '__main__':
    data = torch.zeros(3,3)
    print(data)
    data = data.tolist()
    print(data)
    '''
    tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
    tolist()之后的结果：
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    '''