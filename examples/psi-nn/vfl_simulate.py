import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from termcolor import colored

from linkefl.modelzoo import *
from linkefl.util.progress import progress_bar

# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# import numpy as np
# np.random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')


class VFLModel(nn.Module):
    def __init__(self):
        super(VFLModel, self).__init__()
        self.active_bottom_model = ResNet18(in_channel=3)
        self.passive_bottom_model = ResNet18(in_channel=3)
        self.active_cut_model = CutLayer(10, 10)
        self.passive_cut_model = CutLayer(10, 10)
        self.top_model = MLP([10, 10], activate_input=True, activate_output=False)

    def forward(self, x):
        active_inputs = x[:, :, :16, :]  # first half
        passive_inputs = x[:, :, 16:, :]  # second half
        active_outputs = self.active_cut_model(self.active_bottom_model(active_inputs))
        passive_outputs = self.passive_cut_model(
            self.passive_bottom_model(passive_inputs))
        top_inputs = active_outputs + passive_outputs
        outputs = self.top_model(top_inputs)
        return outputs


vfl_model = VFLModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vfl_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    vfl_model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = vfl_model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct,
                        total))
    print(f"train loss: {train_loss}")


def test(epoch):
    global best_acc
    vfl_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = vfl_model(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (
                         test_loss / (batch_idx + 1), 100. * correct / total, correct,
                         total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print(colored('Best model update.', 'red'))
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 50):
    train(epoch)
    test(epoch)
print(f"best acc: {best_acc}")

# without seed 0: 82.23%
# with seed 0   : 81.83%%