# coding:utf-8
import torch

'''
在pytorch中，只有很少几个操作是不改变tensor的内容本身，而只是重新定义下标与元素的对应关系的
换句话说，这种操作不进行数据拷贝和数据的改变，变得是元数据
这些操作是：
narrow()、view()、expand()和transpose()
'''
'''
举个栗子，在使用transpose()进行转置操作时，pytorch并不会创建新的、转置后的tensor，
而是修改了tensor中的一些属性（也就是元数据），使得此时的offset和stride是与转置tensor相对应的。
转置的tensor和原tensor的内存是共享的！
'''
x = torch.randn(3, 2)
y = x.transpose(0, 1)
x[0, 0] = 233
print(y[0, 0])  # tensor(233.)
print(x)
print(y)
'''
tensor(233.)
tensor([[ 2.3300e+02, -1.1448e+00],
        [ 2.0348e-01,  1.1871e-01],
        [ 9.2045e-02, -3.8829e-01]])
tensor([[ 2.3300e+02,  2.0348e-01,  9.2045e-02],
        [-1.1448e+00,  1.1871e-01, -3.8829e-01]])
也就是说，经过上述操作后得到的tensor，它内部数据的布局方式和从头开始创建一个这样的常规的tensor的布局方式是不一样的！
于是…这就有contiguous()的用武之地了。
'''
print('-' * 100)

'''
在上面的例子中，x是contiguous的，但y不是（因为内部数据不是通常的布局方式）。
注意不要被contiguous的字面意思“连续的”误解，tensor中数据还是在内存中一块区域里，只是布局的问题！

当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一毛一样。

一般来说这一点不用太担心，如果你没在需要调用contiguous()的地方调用contiguous()，运行时会提示你：

RuntimeError: input is not contiguous

只要看到这个错误提示，加上contiguous()就好啦～
'''