import time

from termcolor import colored
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from linkefl.common.const import Const
from linkefl.common.factory import messenger_factory, crypto_factory
from linkefl.splitnn.model import *

class ClientNN:
    def __init__(self, 
                 epochs, 
                 batch_size, 
                 model, 
                 optimizer, 
                 messenger, 
                 crypto_type, 
                 *,
                 precision=0.001, 
                 random_state=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.messenger = messenger
        self.crypto_type = crypto_type
        self.precision = precision
        self.random_state = random_state

    def _init_dataloader(self, dataset, is_shuffle):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=is_shuffle)
        return dataloader

    def train(self, trainset, testset):
        train_dataloader = self._init_dataloader(trainset, True)
        test_dataloader = self._init_dataloader(testset, False)

        self.model.train()
        start_time = time.time()
        for epoch in range(self.epochs):
            for batch_idx, (X, y) in enumerate(train_dataloader):
                outputs = self.model(X)
                self.messenger.send((outputs.data, y))

                grads = self.messenger.recv() # blocking
                self.optimizer.zero_grad()
                outputs.backward(grads)
                self.optimizer.step()
            
            self.messenger.send('train_stop')
            self.validate(testset, existing_loader=test_dataloader)
            is_best = self.messenger.recv()
            if is_best:
                print(colored('Best model updated.', 'red'))
            print(f"Epoch {epoch + 1} finished.\n")
        print(colored('Total training and validation time: {:.2f}'.format(time.time() - start_time), 'red'))

    def validate(self, testset, existing_loader=None):
        if existing_loader is None:
            test_dataloader = self._init_dataloader(testset)
        else:
            test_dataloader = existing_loader

        self.model.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(test_dataloader):
                outputs = self.model(X)
                self.messenger.send((outputs.data, y))
            self.messenger.send('val_stop')


if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'mnist'   # 'cifar10'
    model_name = 'lenet5'    # 'VGG16'
    server_ip, server_port = 'localhost', 20000
    client_ip, client_port = 'localhost', 25000
    _epochs = 20
    _split_layer = 2
    _batch_size = 64
    _learning_rate = 0.05
    _crypto_type = Const.PLAIN

    # 1. Load datasets
    print('Loading dataset...')
    if dataset_name == 'mnist':
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST(root='data', 
                                  train=True, 
                                  download=True, 
                                  transform=transform_train)
        testset = datasets.MNIST(root='data', 
                                 train=False, 
                                 download=True, 
                                 transform=transform_test)
    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose(
                          [transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip(), 
                          transforms.ToTensor(), 
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose(
                         [transforms.ToTensor(), 
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset = datasets.CIFAR10(root='data', 
                                    train=True, 
                                    download=True, 
                                    transform=transform_train)
        testset = datasets.CIFAR10(root='data', 
                                   train=False, 
                                   download=True, 
                                   transform=transform_test)
    else:
        raise ValueError('Invalid dataset name.')
    print('Done.')

    # 2. Create PyTorch model and optimizer
    if model_name == 'lenet5':
        client_model = ClientLenet5(_split_layer)
    elif model_name[0:3] == 'VGG':
        client_model = ClientVGG(model_name, _split_layer)
    else:
        raise ValueError('Invalid model name.')
    _optimizer = torch.optim.SGD(client_model.parameters(), lr=_learning_rate)

    # 3. Initialize messenger
    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.PASSIVE_NAME,
                                   active_ip=server_ip,
                                   active_port=server_port,
                                   passive_ip=client_ip,
                                   passive_port=client_port)

    # 4. Initilize NN protocol and start training
    client_party = ClientNN(epochs=_epochs,
                            batch_size=_batch_size,
                            model=client_model,
                            optimizer=_optimizer,
                            messenger=_messenger,
                            crypto_type=_crypto_type)
    client_party.train(trainset, testset)

    # 5. Close messenger, finish training
    _messenger.close()

