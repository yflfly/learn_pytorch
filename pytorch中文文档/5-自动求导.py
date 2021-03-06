# coding:utf-8
import torch

'''
张量
PyTorch中，所有神经网络的核心是 autograd 包。
autograd 包为张量上的所有操作提供了自动求导机制。它是一个在运行时定义（define-by-run）的框架，这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的。
看例子
torch.Tensor 是这个包的核心类。
    如果设置它的属性 .requires_grad 为 True，那么它将会追踪对于该张量的所有操作。
    当完成计算后可以通过调用 .backward()，来自动计算所有的梯度。
    这个张量的所有梯度将会自动累加到.grad属性.
要阻止一个张量被跟踪历史，可以调用 .detach() 方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。为了防止跟踪历史记录（和使用内存），可以将代码块包装在 with torch.no_grad(): 中。在评估模型时特别有用，因为模型可能具有 requires_grad = True 的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。
还有一个类对于autograd的实现非常重要：Function。Tensor 和 Function 互相连接生成了一个无圈图(acyclic graph)，它编码了完整的计算历史。每个张量都有一个 .grad_fn 属性，该属性引用了创建 Tensor 自身的Function（除非这个张量是用户手动创建的，即这个张量的 grad_fn 是 None ）。
如果需要计算导数，可以在 Tensor 上调用 .backward()。如果 Tensor 是一个标量（即它包含一个元素的数据），则不需要为 backward() 指定任何参数，但是如果它有更多的元素，则需要指定一个 gradient 参数，该参数是形状匹配的张量。
'''

# 创建一个张量并设置requires_grad=True用来追踪其计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)
'''
输出
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
        
'''

# 对张量做一次运算
y = x + 2
print(y)
'''
输出
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
y是计算的结果，所以它有grad_fn属性。
'''

print(y.grad_fn)
# 输出 <AddBackward0 object at 0x7f1b248453c8>

# 对y进行更多操作
z = y * y * 3
out = z.mean()  # 对所有元素求均值
print(z, out)

# 输出
'''
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
'''

# .requires_grad_(...) 原地改变了现有张量的 requires_grad 标志。如果没有指定的话，默认输入的这个标志是 False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# 输出
'''
False
True
<SumBackward0 object at 0x7f1b24845f98>
'''

# 梯度
# 反向传播，因为 out 是一个标量，因此 out.backward() 和 out.backward(torch.tensor(1.)) 等价。
# out.backward() # 自动计算所有的梯度

# 输出导数 d(out)/dx
print(x.grad)

# 输出
'''
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
'''
