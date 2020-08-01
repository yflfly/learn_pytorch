import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)  # x.shape=(120,)
y = np.arange(-6, 6, 0.1)  # y.shape=(120,)
X, Y = np.meshgrid(x, y)  # X.shape=(120, 120), Y.shape=(120,120)

Z = himmelblau([X, Y])
fig = plt.figure("himmeblau")
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

'''
学习网址：
https://www.bilibili.com/video/BV1ht4y1C7GA?p=36
https://www.cnblogs.com/douzujun/p/13300437.html
'''

# 先对x[0],x[1]进行初始化，不同的初始值可能会得到不同的结果
x = torch.tensor([0., 0.], requires_grad=True)
# 定义Adam优化器,指明优化目标是x,学习率是1e-3
optimizer = torch.optim.Adam([x], lr=1e-3)

for step in range(20000):

    pred = himmelblau(x)
    optimizer.zero_grad()  # 将梯度设置为0
    pred.backward()  # 生成当前所在点函数值相关的梯度信息,这里即优化目标的梯度信息
    optimizer.step()  # 使用梯度信息更新优化目标的值,即更新x[0]和x[1]

    if step % 2000 == 0:
        print("step={}，x={}，f(x)={}".format(step, x.tolist(), pred.item()))

''''
Tip：

使用 optimizer的流程就是三行代码：
optimizer.zero_grad()
loss.backward()
optimizer.step()


首先，循环里每个变量都拥有一个优化器
需要在循环里逐个zero_grad()，清理掉上一步的残余值。

之后，对loss调用backward()的时候
它们的梯度信息会被保存在自身的两个属性（grad 和 requires_grad）当中。

最后，调用optimizer.step()，就是一个apply gradients的过程  
将更新值 重新赋给parameters。
'''