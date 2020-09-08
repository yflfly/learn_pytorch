# coding:utf-8
# 导入包和版本查询
import torch
import torch.nn as nn
import torchvision

print(torch.__version__)
print(torch.version.cuda)  # cuda版本查询
print(torch.backends.cudnn.version())  # cudnn版本查询
# print(torch.cuda.get_device_name(0))  # 设备名
