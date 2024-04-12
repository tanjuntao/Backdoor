import torch.nn as nn
import torch.nn.functional as F

from .passport import ConvPassportBlock


class BasicBlock(nn.Module):
    expansion = 1  # for increasing channel

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )  # when stride equals 2, the resulution will drop to half of the original
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            self.expansion * planes,  # block output channel
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        # the first condition means that the image resulution has changed;
        # the second condition means that the input channel is different from the output
        # channel
        # both these two cases need to apply one 1x1 conv2d on the shortcut connection
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FedPassBasicBlock(nn.Module):
    expansion = 1  # for increasing channel

    def __init__(
        self, in_planes, planes, stride=1, loc=-1.0, scale=1.0, passport_mode="multi"
    ):
        super(FedPassBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_passport = ConvPassportBlock(
            in_channels=planes,
            out_channels=self.expansion * planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            loc=loc,
            scale=scale,
            passport_mode=passport_mode,
            activation=None,
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.conv_passport(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4  # for increasing channel

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, bias=False
        )  # stride defaults to 1
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )  # when stride equals 2, the resulution will drop to half of the original
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, out.size()[2:])  # the resulution now is 4x4
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


def ResNet18(in_channel, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channel, num_classes)


def ResNet34(in_channel, num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channel, num_classes)


def ResNet50(in_channel, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channel, num_classes)


def ResNet101(in_channel, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channel, num_classes)


def ResNet152(in_channel, num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], in_channel, num_classes)


class FedPassResNet(nn.Module):
    def __init__(
        self,
        blocks,
        num_blocks,
        in_channel,
        num_classes=10,
        loc=-1.0,
        scale=1.0,
        passport_mode="multi",
    ):
        super(FedPassResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(blocks[0], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[0], 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[0], 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(blocks[0], 512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(
            blocks[1],
            512,
            num_blocks[4],
            stride=1,
            loc=loc,
            scale=scale,
            passport_mode=passport_mode,
        )
        self.linear = nn.Linear(512 * blocks[0].expansion, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # out = F.avg_pool2d(out, out.size()[2:])  # the resulution now is 4x4
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        loc=None,
        scale=None,
        passport_mode=None,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if block == BasicBlock:
                layers.append(block(self.in_planes, planes, stride))
            elif block == FedPassBasicBlock:
                layers.append(
                    block(self.in_planes, planes, stride, loc, scale, passport_mode)
                )
            else:
                pass
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


def FedPassResNet18(
    in_channel, num_classes=10, loc=-1.0, scale=1.0, passport_mode="multi"
):
    return FedPassResNet(
        [BasicBlock, FedPassBasicBlock],
        [2, 2, 2, 1, 1],
        in_channel,
        num_classes,
        loc,
        scale,
        passport_mode,
    )
