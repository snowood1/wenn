from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LeNet', 'lenet5']


# in_channels =1, dim= 800
# in_channels = 3,  dim = 1250

class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_class=10, dim=800):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, 5)  # MNIST add padding=2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(dim, 500)
        self.fc2 = nn.Linear(500, num_class)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def lenet5(in_channels=1):
    model = LeNet(in_channels)
    return model
