import torch.nn as nn
import torch.nn.functional as F

from linkefl.modelzoo.util import TorchModuleType, make_nn_module

from .passport import ConvPassportBlock


class LeNet(nn.Module):
    def __init__(
        self,
        in_channel,
        num_classes=10,
        activation: TorchModuleType = "ReLU",
    ):
        super(LeNet, self).__init__()
        self.activation = make_nn_module(activation)
        self.num_classes = num_classes
        # input size should be [in_channel, 32, 32]
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(80, 120)  # half image
        # self.fc1 = nn.Linear(400, 120)   # full image
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.activation((self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.activation((self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.activation((self.fc1(out)))
        out = self.activation((self.fc2(out)))
        out = self.fc3(out)  # logits
        return out


class FedPassLeNet(nn.Module):
    def __init__(
        self,
        in_channel,
        num_classes=10,
        activation: TorchModuleType = "ReLU",
        loc=-1.0,
        scale=1.0,
        passport_mode="multi",
    ):
        super(FedPassLeNet, self).__init__()
        self.activation = make_nn_module(activation)
        self.num_classes = num_classes
        # input size should be [in_channel, 32, 32]
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = ConvPassportBlock(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True,
            loc=loc,
            scale=scale,
            passport_mode=passport_mode,
            activation=None,
        )
        self.fc1 = nn.Linear(80, 120)  # half image
        # self.fc1 = nn.Linear(400, 120)   # full image
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.activation((self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.activation((self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.activation((self.fc1(out)))
        out = self.activation((self.fc2(out)))
        out = self.fc3(out)  # logits
        return out
