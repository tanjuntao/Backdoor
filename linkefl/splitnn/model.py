import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

'''
The following classes are designed to split widely used models for two-party and u-shape split learning. We can split models into different parts and allocate them to the server and the client.
The parameters _in_layer and _out_layer are the splitting strategy.

e.g.
If a model has 5 layers, we define that it has 6 split layers including (0, 1, 2, 3, 4, 5).
if we want to extract layer2 and layer3 of the model, the parameter _in_layer = 1 and _out_layer = 3; If we want to extract layer1 to layer4 of the model, the parameter _in_layer = 0 and _out_layer = 4.
'''

class SplitLenet5(nn.Module):
    def __init__(self, _in_layer=0, _out_layer=5):
        super(SplitLenet5, self).__init__()

        self.in_layer = _in_layer
        self.out_layer = _out_layer
        if self.in_layer not in [0, 1, 2, 3, 4]:
            raise ValueError('Invalid in layer.')             
        if self.out_layer not in [1, 2, 3, 4, 5]:
            raise ValueError('Invalid out layer.')
        if self.out_layer <= self.in_layer:
            raise ValueError('Invalid in layer and out layer: out layer should be larger than in layer.')

        if self.in_layer == 0:
            self.layer1 = nn.Sequential(
                          nn.Conv2d(1, 6, kernel_size=5,padding=2), 
                          nn.ReLU(False), 
                          nn.MaxPool2d(kernel_size=2, stride=2))
        if self.in_layer <= 1 and self.out_layer >= 2: 
            self.layer2 = nn.Sequential(
                          nn.Conv2d(6, 16, kernel_size=5), 
                          nn.ReLU(False), 
                          nn.MaxPool2d(kernel_size=2, stride=2))
        if self.in_layer <= 2 and self.out_layer >= 3:
            self.layer3 = nn.Sequential(
                          nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), 
                          nn.Sigmoid())
        if self.in_layer <= 3 and self.out_layer >= 4:
            self.layer4 = nn.Sequential(
                          nn.Linear(120, 84), 
                          nn.Sigmoid())
        if self.out_layer == 5:
            self.layer5 = nn.Linear(84, 10)

    def forward(self, x):
        if self.in_layer == 0: x = self.layer1(x)
        if self.out_layer == 1: return x
        if self.in_layer in [0, 1]: x = self.layer2(x)
        if self.out_layer == 2: return x
        if self.in_layer in [0, 1, 2]: x = self.layer3(x)
        if self.out_layer == 3: return x
        if self.in_layer in [0, 1, 2, 3]: x = self.layer4(x)
        if self.out_layer == 4: return x
        x = self.layer5(x)
        return x

cfg_split = {
    'VGG11': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'VGG13': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'VGG16': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'VGG19': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']],}

class SplitVGG(nn.Module):
    def __init__(self, _model_name, _in_layer=0, _out_layer=6):
        super(SplitVGG, self).__init__()
        self.model_name = _model_name
        self.in_layer = _in_layer
        self.out_layer = _out_layer
        if self.in_layer not in [0, 1, 2, 3, 4, 5]:
            raise ValueError('Invalid in layer.')             
        if self.out_layer not in [1, 2, 3, 4, 5, 6]:
            raise ValueError('Invalid out layer.')
        if self.out_layer <= self.in_layer:
            raise ValueError('Invalid in layer and out layer: out layer should be larger than in layer.')

        if self.in_layer == 0:
            self.layer1 = self._make_layers(cfg_split[self.model_name][0], 3)
        if self.in_layer <= 1 and self.out_layer >= 2: 
            self.layer2 = self._make_layers(cfg_split[self.model_name][1], 64)
        if self.in_layer <= 2 and self.out_layer >= 3: 
            self.layer3 = self._make_layers(cfg_split[self.model_name][2],128)
        if self.in_layer <= 3 and self.out_layer >= 4: 
            self.layer4 = self._make_layers(cfg_split[self.model_name][3],256)
        if self.in_layer <= 4 and self.out_layer >= 5: 
            self.layer5 = self._make_layers(cfg_split[self.model_name][4],512)
        if self.out_layer == 6:
            self.AvgPool = nn.AvgPool2d(kernel_size=1, stride=1)
            self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        if self.in_layer == 0: x = self.layer1(x)
        if self.out_layer == 1: return x
        if self.in_layer in [0, 1]: x = self.layer2(x)
        if self.out_layer == 2: return x
        if self.in_layer in [0, 1, 2]: x = self.layer3(x)
        if self.out_layer == 3: return x
        if self.in_layer in [0, 1, 2, 3]: x = self.layer4(x)
        if self.out_layer == 4: return x
        if self.in_layer in [0, 1, 2, 3, 4]: x = self.layer5(x)
        if self.out_layer == 5: return x
        x = self.AvgPool(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return x

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
