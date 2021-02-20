# coding:utf-8
import torch

'''
有时候需要指定比现有模块序列更复杂的模型；对于这些情况，可以通过继承 nn.Module 并定义
forward 函数，这个 forward 函数可以 使用其他模块或者其他的自动求导运算来接收输入
tensor，产生输出tensor。
'''


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们实例化了两个nn.Linear模块，并将它们作为成员变量。
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在前向传播的函数中，我们接收一个输入的张量，也必须返回一个输出张量。
        我们可以使用构造函数中定义的模块以及张量上的任意的（可微分的）操作。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N是批大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出的随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 通过实例化上面定义的类来构建我们的模型。
model = TwoLayerNet(D_in, H, D_out)

# 构造损失函数和优化器。
# SGD构造函数中对model.parameters()的调用，
# 将包含模型的一部分，即两个nn.Linear模块的可学习参数。
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(10):
    # 前向传播：通过向模型传递x计算预测值y
    y_pred = model(x)
    # 计算并输出loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    # 清零梯度，反向传播，更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''
输出结果如下所示：
0 659.6338500976562
1 607.0213623046875
2 561.9886474609375
3 523.0420532226562
4 488.556884765625
5 457.31097412109375
6 428.973388671875
7 402.9942626953125
8 378.9892578125
9 356.83355712890625
'''
