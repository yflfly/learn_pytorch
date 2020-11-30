# coding:utf-8
import torch

'''
torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)

解释：
沿给定dim维度返回输入张量input中 k 个最大值。
如果不指定dim，则默认为input的最后一维。
如果为largest为 False ，则返回最小的 k 个值。

返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标。
如果设定布尔值sorted 为_True_，将会确保返回的 k 个值被排序。

参数：
input (Tensor) – 输入张量
k (int) – “top-k”中的k
dim (int, optional) – 排序的维
largest (bool, optional) – 布尔值，控制返回最大或最小值
sorted (bool, optional) – 布尔值，控制返回值是否排序
out (tuple, optional) – 可选输出张量 (Tensor, LongTensor) output buffer
'''


def h1():
    output = torch.tensor([[-5.4783, 0.2298],
                           [-4.2573, -0.4794],
                           [-0.1070, -5.1511],
                           [-0.1785, -4.3339]])
    maxk = max((1,))  # 取top1准确率，若取top1和top5准确率改为max((1,5))
    _, pred = output.topk(maxk, 1, True, True)
    output_topk = torch.topk(output, 1).indices.squeeze(0).tolist()
    # topk参数中，maxk取得是top1准确率，dim=1是按行取值， largest=1是取最大值
    print(maxk)  # 1
    print(_)
    print(pred)
    print('*'*100)
    print(output_topk)  # [[1], [1], [0], [0]]
    '''
    tensor([[ 0.2298],
        [-0.4794],
        [-0.1070],
        [-0.1785]])
    tensor([[1],
            [1],
            [0],
            [0]])
    '''
    '''
    _是top1的值，pred是最大值的索引（size=4*1），一般会进行转置处理同真实值对比
    '''


if __name__ == '__main__':
    h1()
