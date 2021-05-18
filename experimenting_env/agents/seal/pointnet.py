from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class PCDFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 2, 3)
        self.conv2 = torch.nn.Conv3d(2, 2, 3)
        self.conv3 = torch.nn.Conv2d(2, 4, 3)
        self.conv4 = torch.nn.Conv2d(4, 8, 3)
        self.conv5 = torch.nn.Conv2d(8, 16, 3)
        self.conv6 = torch.nn.Conv3d(16, 16, 3)
        self.fc1 = nn.Linear(3092544, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = F.avg_pool2d(x.flatten(-2), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = x.flatten(1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
