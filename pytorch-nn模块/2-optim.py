# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
到目前为止，我们已经通过手动改变包含可学习参数的张量来更新模型的权重。对于随机梯度下
降(SGD/stochastic gradient descent)等简单的优化算法来说，这不是一个很大的负担，但在实践
中，我们经常使用AdaGrad、RMSProp、Adam等更复杂的优化器来训练神经网络。
'''

# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10
# 创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# 使用nn包将我们的模型定义为一系列的层。
# nn.Sequential是包含其他模块的模块，并按顺序应用这些模块来产生其输出。
# 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量。
# 在构造模型之后，我们使用.to()方法将其移动到所需的设备。
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# nn包还包含常用的损失函数的定义；
# 在这种情况下，我们将使用平均平方误差(MSE)作为我们的损失函数。
# 设置reduction='sum'，表示我们计算的是平方误差的“和”，而不是平均值;
# 这是为了与前面我们手工计算损失的例子保持一致，
# 但是在实践中，通过设置reduction='elementwise_mean'来使用均方误差作为损失更为常见。
loss_fn = torch.nn.MSELoss(reduction='sum')

# 使用optim包定义优化器（Optimizer）。Optimizer将会为我们更新模型的权重。
# 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法。
# Adam构造函数的第一个参数告诉优化器应该更新哪些张量。
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10):
    # 前向传播：通过像模型输入x计算预测的y
    y_pred = model(x)
    # 计算并打印loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
    optimizer.zero_grad()
    # 反向传播：根据模型的参数计算loss的梯度
    loss.backward()
    # 调用Optimizer的step函数使它所有参数更新
    optimizer.step()
'''
输出结果如下所示：
0 645.4808349609375
1 628.6141967773438
2 612.242431640625
3 596.3660888671875
4 581.0346069335938
5 566.214599609375
6 551.9215698242188
7 538.0454711914062
8 524.5087280273438
9 511.39337158203125
'''