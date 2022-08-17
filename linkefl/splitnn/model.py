import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

class ClientLenet5(nn.Module):
    def __init__(self, _split_layer):
        super(ClientLenet5, self).__init__()
        self.split_layer = _split_layer
        if self.split_layer not in [1, 2, 3, 4]:
            raise ValueError('Invalid split layer.')
        self.layer1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5,padding=2), 
                                    nn.ReLU(False), nn.MaxPool2d(kernel_size=2, stride=2))
        if self.split_layer >= 2: 
            self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5), 
                          nn.ReLU(False), nn.MaxPool2d(kernel_size=2, stride=2))
        if self.split_layer >= 3: 
            self.layer3 = nn.Sequential(nn.Flatten(),nn.Linear(16 * 5 * 5, 120), 
                                        nn.Sigmoid())
        if self.split_layer >= 4:
            self.layer4 = nn.Sequential(nn.Linear(120, 84), nn.Sigmoid())

    def forward(self, x): 
        x = self.layer1(x)
        if self.split_layer == 1: return x
        x = self.layer2(x)
        if self.split_layer == 2: return x
        x = self.layer3(x)
        if self.split_layer == 3: return x
        x = self.layer4(x)
        if self.split_layer == 4: return x
        exit(0)
        
class ServerLenet5(nn.Module):
    def __init__(self, _split_layer):
        super(ServerLenet5, self).__init__()
        self.split_layer = _split_layer
        if self.split_layer not in [1, 2, 3, 4]:
            raise ValueError('Invalid split layer.')
        if self.split_layer <= 2: 
            self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5), 
                          nn.ReLU(False), nn.MaxPool2d(kernel_size=2, stride=2))
        if self.split_layer <= 3:
            self.layer3 = nn.Sequential(nn.Flatten(),nn.Linear(16 * 5 * 5, 120), 
                                        nn.Sigmoid())
        if self.split_layer <= 4:
            self.layer4 = nn.Sequential(nn.Linear(120, 84), nn.Sigmoid())
        self.layer5 = nn.Linear(84, 10)

    def forward(self, x):
        if self.split_layer == 1: x = self.layer2(x)
        if self.split_layer in [1,2]: x = self.layer3(x)
        if self.split_layer in [1,2,3]: x = self.layer4(x)
        x=self.layer5(x)
        return x

cfg_split = {
    'VGG11': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'VGG13': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'VGG16': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'VGG19': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']],}

class VGG(nn.Module):
    def __init__(self, _model_name):
        super(VGG, self).__init__()
        self.model_name = _model_name
        self.layer1 = self._make_layers(cfg_split[self.model_name][0], 3)
        self.layer2 = self._make_layers(cfg_split[self.model_name][1], 64)
        self.layer3 = self._make_layers(cfg_split[self.model_name][2],128)
        self.layer4 = self._make_layers(cfg_split[self.model_name][3],256)
        self.layer5 = self._make_layers(cfg_split[self.model_name][4],512)
        self.AvgPool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.AvgPool(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

class ClientVGG(nn.Module):
    def __init__(self, _model_name, _split_layer):
        super(ClientVGG, self).__init__()
        self.split_layer = _split_layer
        self.model_name = _model_name
        self.layer1 = self._make_layers(cfg_split[self.model_name][0], 3)
        if self.split_layer>=2: 
            self.layer2 = self._make_layers(cfg_split[self.model_name][1], 64)
        if self.split_layer>=3: 
            self.layer3 = self._make_layers(cfg_split[self.model_name][2],128)
        if self.split_layer>=4: 
            self.layer4 = self._make_layers(cfg_split[self.model_name][3],256)

    def forward(self, x):
        x = self.layer1(x)
        if self.split_layer==1: return x
        x = self.layer2(x)
        if self.split_layer==2: return x
        x = self.layer3(x)
        if self.split_layer==3: return x
        x = self.layer4(x)
        if self.split_layer==4: return x
        exit(0)

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

class ServerVGG(nn.Module):
    def __init__(self, _model_name, _split_layer):
        super(ServerVGG, self).__init__()
        self.model_name = _model_name
        self.split_layer = _split_layer
        if self.split_layer not in [1,2,3,4]: exit(0)
        if self.split_layer<=1: 
            self.layer2 = self._make_layers(cfg_split[self.model_name][1], 64)
        if self.split_layer<=2: 
            self.layer3 = self._make_layers(cfg_split[self.model_name][2],128)
        if self.split_layer<=3: 
            self.layer4 = self._make_layers(cfg_split[self.model_name][3],256)
        self.layer5 = self._make_layers(cfg_split[self.model_name][4],512)
        self.AvgPool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        if self.split_layer == 1: x = self.layer2(x)
        if self.split_layer in [1,2]: x = self.layer3(x)
        if self.split_layer in [1,2,3]: x = self.layer4(x)
        x = self.layer5(x)
        x = self.AvgPool(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
