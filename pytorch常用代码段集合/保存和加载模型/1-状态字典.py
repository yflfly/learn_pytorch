# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义模型
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化模型
model = TheModelClass()

# 初始化优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 打印模型的状态字典
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

'''
输出结果如下所示：
Model's state_dict:
conv1.weight 	 torch.Size([6, 3, 5, 5])
conv1.bias 	 torch.Size([6])
conv2.weight 	 torch.Size([16, 6, 5, 5])
conv2.bias 	 torch.Size([16])
fc1.weight 	 torch.Size([120, 400])
fc1.bias 	 torch.Size([120])
fc2.weight 	 torch.Size([84, 120])
fc2.bias 	 torch.Size([84])
fc3.weight 	 torch.Size([10, 84])
fc3.bias 	 torch.Size([10])
Optimizer's state_dict:
state 	 {}
param_groups 	 [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [2091698046080, 2091698046152, 2091698046224, 2091698046296, 2091698046368, 2091698046440, 2091698046512, 2091698046584, 2091698046728, 2091698046800]}]

'''

'''
讲解：
在PyTorch中， torch.nn.Module 模型的可学习参数（即权重和偏差）包含在模型的参数中，（使 用 model.parameters() 可以进行访问）。 
state_dict 是Python字典对象，它将每一层映射到其参数张量。注意，只有具有可学习参数的层（如卷积层，线性层等）的模型 才具有 state_dict 这一项。
目标优化 torch.optim 也有 state_dict 属性，它包含有关优化器的状态信息，以及使用的超参数。

因为state_dict的对象是Python字典，所以它们可以很容易的保存、更新、修改和恢复，为PyTorch模型和优化器添加了大量模块。
'''
