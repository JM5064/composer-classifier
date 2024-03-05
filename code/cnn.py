import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):

    def __init__(self, num_channels, classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding="same")
        self.linear = nn.Linear(180, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x
