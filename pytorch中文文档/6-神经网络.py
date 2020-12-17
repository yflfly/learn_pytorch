# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
1)使用torch.nn包来构建神经网络
2)我们已经学习了autograd，nn包则依赖于autograd包来定义模型并对它们求导。
一个nn.Module包含各个层和一个forward(input)方法，该方法返回output。
3)一个神经网络的典型训练过程如下：
    定义包含一些可学习参数（或者叫权重）的神经网络
    在输入数据集上迭代
    通过网络处理输入
    计算损失（输出和正确答案的距离）
    将梯度反向传播给网络的参数
    更新网络的权重，一般使用一个简单的规则：weight = weight - learning_rate * gradient
'''


# 1、定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # 见下面该函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去batch_size的其他所有维度,pytorch中为[batch_size,channle,h,w]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# 输出
'''
Net(
    (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
    (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (fc1): Linear(in_features=400, out_features=120, bias=True)
    (fc2): Linear(in_features=120, out_features=84, bias=True)
    (fc3): Linear(in_features=84, out_features=10, bias=True)
)
我们只需要定义 forward 函数，backward函数会在使用autograd时自动定义，backward函数用来计算导数。
可以在 forward 函数中使用任何针对张量的操作和计算。

一个模型的可学习参数可以通过net.parameters()返回
'''

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
'''
输出：
10
torch.Size([6, 1, 5, 5])
'''

# 尝试一个随机的32x32的输入：
input = torch.randn(1, 1, 32, 32)  # 1分别代表batch_size,channle
out = net(input)
print(out)
'''
# 输出：上网络结果最后输出10个分类
tensor([[ 0.0325, -0.0515,  0.0057, -0.0461,  0.0948, -0.1213,  0.0766,  0.1587,
          0.0357,  0.0935]], grad_fn=<AddmmBackward>)
'''
# 清零所有参数的梯度缓存，然后进行随机梯度的反向传播：
net.zero_grad()
out.backward(torch.randn(1, 10))
'''
注意：
torch.nn只支持小批量处理（mini-batches）。整个torch.nn包只支持小批量样本的输入，不支持单个样本。
比如，nn.Conv2d 接受一个4维的张量，即nSamples x nChannels x Height x Width
如果是一个单独的样本，只需要使用input.unsqueeze(0)来添加一个“假的”批大小维度。
'''
# 损失函数
'''
一个损失函数接受一对(output, target)作为输入，计算一个值来估计网络的输出和目标值相差多少。
nn包中有很多不同的损失函数。nn.MSELoss是比较简单的一种，它计算输出和目标的均方误差（mean-squared error）。
'''
output = net(input)
target = torch.randn(10)  # 本例子中使用模拟数据
target = target.view(1, -1)  # 使目标值与数据值形状一致
criterion = nn.MSELoss()  # 均方误差函数

loss = criterion(output, target)
print(loss)
'''
输出：
tensor(1.5852, grad_fn=<MseLossBackward>)
'''

'''
现在，如果使用loss的.grad_fn属性跟踪反向传播过程，会看到计算图如下：
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
所以，当我们调用loss.backward()，整张图开始关于loss微分，
图中所有设置了requires_grad=True的张量的.grad属性累积着梯度张量。
'''
# 反向传播
'''
我们只需要调用loss.backward()来反向传播权重。我们需要清零现有的梯度，否则梯度将会与已有的梯度累加。
现在，我们将调用loss.backward()，并查看conv1层的偏置（bias）在反向传播前后的梯度。
'''
net.zero_grad()  # 清零所有参数（parameter）的梯度缓存
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
'''
输出：
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([-0.0056,  0.0004, -0.0066, -0.0058,  0.0063,  0.0057])
'''
# 更新权重
'''
最简单的更新规则是随机梯度下降法（SGD）:
weight = weight - learning_rate * gradient

在使用神经网络时，可能希望使用各种不同的更新规则，如SGD、Nesterov-SGD、Adam、RMSProp等。
为此，我们构建了一个较小的包torch.optim，它实现了所有的这些方法。使用它很简单：
'''
import torch.optim as optim

# 创建优化器（optimizer）
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练的迭代中：
optimizer.zero_grad()  # 清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # 更新参数
