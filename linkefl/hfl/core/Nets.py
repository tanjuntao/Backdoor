import torch
import torch.nn.functional as F
from torch import nn




class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, num_channels=1,size = 320):
        super(SimpleCNN, self).__init__()
        self.size = size
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out

class SimpleCNNFMnist(nn.Module):
    def __init__(self, num_classes=10, num_channels=1,size = 320):
        super(SimpleCNNFMnist, self).__init__()
        self.size = size
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out

class SimpleCNNCifar(nn.Module):
    def __init__(self, num_classes=10, num_channels=1,size = 320):
        super(SimpleCNNCifar, self).__init__()
        self.size = size
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out

node_1 = 10
node_2 = 10

class MLP(nn.Module):
    def __init__(self,num_classes,in_features):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, node_1)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(node_1, node_2)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(node_2, num_classes)

    def forward(self,x):
        x = x.to(torch.float32)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

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
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
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
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, num_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_channels)


def ResNet34(num_classes=10, num_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, num_channels)


def ResNet50(num_classes=10, num_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, num_channels)


def ResNet101(num_classes=10, num_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, num_channels)


def ResNet152(num_classes=10, num_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, num_channels)


class LogReg(nn.Module):
    def __init__(self, in_features=2, out_feautes=2):
        super(LogReg, self).__init__()
        self.features = nn.Linear(in_features, out_features=out_feautes)
        # self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)
        # out = self.sigmoid(x)
        # out = out.squeeze(-1)
        return x
#
# class LeNet(nn.Module):
#     def __init__(self,num_classes,num_channels):
#         super(LeNet, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(num_channels, 6, 5), # in_channels, out_channels, kernel_size
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2), # kernel_size, stride
#             nn.Conv2d(6, 16, 5),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(16*4*4, 120),
#             nn.Sigmoid(),
#             nn.Linear(120, 84),
#             nn.Sigmoid(),
#             nn.Linear(84, num_classes)
#         )
#
#     def forward(self, img):
#         feature = self.conv(img)
#         output = self.fc(feature.view(img.shape[0], -1))
#         return output

class LeNet(nn.Module):
    def __init__(self,num_classes,num_channels):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(16 * 5 * 5, 120),
        #     nn.ReLU()
        # )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        size = 1
        for i in range(1,len(x.size())):
            size = size*x.size()[i]
        fc1 = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU()
        )
        # x = x.view(size, -1)
        x = x.view(-1, size)

        x = fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class LeNet_MNIST(nn.Module):
    def __init__(self,num_classes,num_channels):
        super(LeNet_MNIST, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LeNet_MNIST(nn.Module):
    def __init__(self,num_classes,num_channels):
        super(LeNet_MNIST, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def Nets(model_name, num_classes, data_name = "MNIST",num_channels=1,in_features=10):

    model_dict = {"CNNmnist":SimpleCNN,"CNNfashion_mnist":SimpleCNNFMnist,"CNNcifar10":SimpleCNNCifar,
                  "LeNetmnist":SimpleCNN,"LeNetfashion_mnist":SimpleCNNFMnist,"LeNetcifar10":SimpleCNNCifar,
                  "ResNet18mnist":ResNet18,"ResNet18Fashionmnist":ResNet18,"ResNet18cifar10":ResNet18,}
    model = model_dict[model_name+data_name]
    return model(num_classes,num_channels)

    # if model_name == "CNN":
    #     return SimpleCNN(num_classes, num_channels)
    # elif model_name == "ResNet18":
    #     return ResNet18(num_classes, num_channels)
    # elif model_name == "LeNet":
    #     return LeNet(num_classes, num_channels)


class LinReg(nn.Module):
    def __init__(self,in_features):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.linear(x)
        return out

