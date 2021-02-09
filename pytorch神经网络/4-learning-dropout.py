# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutFC(nn.Module):
    def __init__(self):
        super(DropoutFC, self).__init__()
        self.fc = nn.Linear(100, 20)
        self.training = True

    def forward(self, input):
        out = self.fc(input)
        out = F.dropout(out, p=0.5, training=self.training)
        return out


Net = DropoutFC()
Net.train()

# train the Net
