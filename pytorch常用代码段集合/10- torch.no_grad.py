# coding:utf-8
import torch

'''
requires_grad
在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
tensor的requires_grad的属性默认为False,若一个节点（叶子变量：自己创建的tensor）requires_grad被设置为True，
那么所有依赖它的节点requires_grad都为True（即使其他相依赖的tensor的requires_grad = False）
'''


def h1():
    x = torch.randn(10, 5, requires_grad=True)
    y = torch.randn(10, 5, requires_grad=False)
    z = torch.randn(10, 5, requires_grad=False)
    w = x + y + z
    print(w.requires_grad)  # True


'''
with torch.no_grad：
即使一个tensor（命名为x）的requires_grad = True，
由x得到的新tensor（命名为w-标量）requires_grad也为False，且grad_fn也为None,即不会对w求导

'''


def h2():
    x = torch.randn(10, 5, requires_grad=True)
    y = torch.randn(10, 5, requires_grad=True)
    z = torch.randn(10, 5, requires_grad=True)
    with torch.no_grad():
        w = x + y + z
        print(w.requires_grad)  # False
        print(w.grad_fn)  # None grad_fn：指向Function对象，用于反向传播的梯度计算之用
    print(w.requires_grad)  # False


if __name__ == '__main__':
    h1()
    h2()
