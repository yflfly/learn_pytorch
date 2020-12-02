# coding:utf-8
import torch
'''
torch.optim是一个实现了各种优化算法的库。
大部分常用的方法得到支持，并且接口具备足够的通用性，使得未来能够集成更加复杂的方法。
如何使用optimizer
为了使用torch.optim，你需要构建一个optimizer对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。

'''
# 构建
'''
为了构建一个Optimizer，你需要给它一个包含了需要优化的参数（必须都是Variable对象）的iterable。
然后，你可以设置optimizer的参 数选项，比如学习率，权重衰减，等等。

例子：
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)

'''

'''
所有的optimizer都实现了step()方法，这个方法会更新所有的参数。它能按两种方式来使用：

optimizer.step()

这是大多数optimizer所支持的简化版本。一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数。

例子:
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    
'''