# coding:utf-8
import torch

'''
一个Torch张量与一个NumPy数组的转换很简单
Torch张量和NumPy数组将共享它们的底层内存位置，因此当一个改变时,另外也会改变。
'''


def h_1():
    a = torch.ones(5)
    print(a)

    # 输出
    # tensor([1., 1., 1., 1., 1.])

    b = a.numpy()
    print(b)

    # 输出
    # [1. 1. 1. 1. 1.]

    a.add_(1)
    print(a)
    print(b)

    # 输出
    # tensor([2., 2., 2., 2., 2.])
    # [2. 2. 2. 2. 2.]


def h_2():  # numpy数组转换为张量
    import numpy as np
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)

    # 输出
    '''
    [2. 2. 2. 2. 2.]
    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    '''

    '''
    CPU上的所有张量(CharTensor除外)都支持与Numpy的相互转换。
    张量可以使用.to方法移动到任何设备（device）上：
    '''
    # 当GPU可用时,我们可以运行以下代码
    # 我们将使用`torch.device`来将tensor移入和移出GPU
    x = torch.rand(5, 3)
    if torch.cuda.is_available():
        device = torch.device("cuda")  # a CUDA device object
        y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
        x = x.to(device)  # 或者使用`.to("cuda")`方法
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))  # `.to`也能在移动时改变dtype

    # 输出
    # tensor([1.0445], device='cuda:0')
    # tensor([1.0445], dtype=torch.float64)


if __name__ == '__main__':
    h_2()
