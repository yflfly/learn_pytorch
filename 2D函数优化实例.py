import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)              # x.shape=(120,)
y = np.arange(-6, 6, 0.1)              # y.shape=(120,)
X, Y = np.meshgrid(x, y)               # X.shape=(120, 120), Y.shape=(120,120)

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