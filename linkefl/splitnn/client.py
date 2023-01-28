import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from termcolor import colored
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.splitnn.model import SplitLenet5, SplitVGG


class ClientNN:
    def __init__(
        self,
        epochs,
        batch_size,
        bottom_model,
        bottom_opt,
        messenger,
        crypto_type,
        u_shape_flag,
        *,
        top_model=None,
        top_opt=None,
        loss_fn=None,
        precision=0.001,
        random_state=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.bottom_model = bottom_model
        self.bottom_opt = bottom_opt
        self.top_model = top_model
        self.top_opt = top_opt
        self.loss_fn = loss_fn
        self.messenger = messenger
        self.crypto_type = crypto_type
        self.u_shape_flag = u_shape_flag
        self.precision = precision
        self.random_state = random_state

    def _init_dataloader(self, dataset, is_shuffle):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=is_shuffle)
        return dataloader

    def train(self, trainset, testset):
        train_dataloader = self._init_dataloader(trainset, True)
        test_dataloader = self._init_dataloader(testset, False)
        if self.u_shape_flag:
            self._u_shape_train(train_dataloader, test_dataloader)
        else:
            self._two_party_train(train_dataloader, test_dataloader)

    def _u_shape_train(self, trainloader, testloader):
        start_time = time.time()
        best_acc, best_auc = 0, 0
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch + 1))
            self._u_shape_train_per_epoch(trainloader)
            curr_acc, curr_auc = self._u_shape_val_per_epoch(testloader)
            is_best = False
            if curr_acc > best_acc:
                print(colored("Best model update.\n", "red"))
                is_best, best_acc = True, curr_acc
            if curr_auc > best_auc:
                best_auc = curr_auc
            self.messenger.send(is_best)
        print(
            colored(
                "Total training and validation time: {:.2f}".format(
                    time.time() - start_time
                ),
                "red",
            )
        )
        print(colored("Best testing accuracy: {:.5f}".format(best_acc), "red"))
        print(colored("Best testing auc: {:.5f}".format(best_auc), "red"))

    def _two_party_train(self, trainloader, testloader):
        start_time = time.time()
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch + 1))
            self._two_party_train_per_epoch(trainloader)
            self._two_party_val_per_epoch(testloader)
            is_best = self.messenger.recv()
            if is_best:
                print(colored("Best model updated.", "red"))
            print(f"Epoch {epoch + 1} finished.\n")
        print(
            colored(
                "Total training and validation time: {:.2f}".format(
                    time.time() - start_time
                ),
                "red",
            )
        )

    def _u_shape_train_per_epoch(self, trainloader):
        self.bottom_model.train()
        self.top_model.train()
        for batch_idx, (X, y) in enumerate(trainloader):
            self.bottom_opt.zero_grad()
            smashed_data1 = self.bottom_model(X)
            self.messenger.send(smashed_data1.data)

            smashed_data2 = self.messenger.recv()
            smashed_data2.requires_grad_()
            self.top_opt.zero_grad()
            preds = self.top_model(smashed_data2)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.top_opt.step()
            self.messenger.send(smashed_data2.grad)

            bottom_grads = self.messenger.recv()
            smashed_data1.backward(bottom_grads)
            self.bottom_opt.step()
        self.messenger.send("train_stop")

    def _u_shape_val_per_epoch(self, testloader):
        num_batches, num_samples, test_loss, correct = 0, 0, 0, 0
        labels, probs = np.array([]), np.array([])  # used for computing AUC score
        self.bottom_model.eval()
        self.top_model.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(testloader):
                smahsed_data1 = self.bottom_model(X)
                self.messenger.send(smahsed_data1.data)

                smashed_data2 = self.messenger.recv()
                pred = self.top_model(smashed_data2)

                labels = np.append(labels, y.numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(pred[:, 1]).numpy())
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                num_batches += 1
                num_samples += y.size(0)

            test_loss /= num_batches
            acc = correct / num_samples
            n_classes = len(np.unique(labels))
            if n_classes == 2:
                auc = roc_auc_score(labels, probs)
            else:
                auc = 0
            print(
                f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%,"
                f" Auc: {(100 * auc):>0.2f}%,"
                f" Avg loss: {test_loss:>8f}"
            )
            self.messenger.send("val_stop")
            return acc, auc

    def _two_party_train_per_epoch(self, trainloader):
        self.bottom_model.train()
        for batch_idx, (X, y) in enumerate(trainloader):
            self.bottom_opt.zero_grad()
            smashed_data1 = self.bottom_model(X)
            self.messenger.send((smashed_data1.data, y))
            bottom_grads = self.messenger.recv()
            smashed_data1.backward(bottom_grads)
            self.bottom_opt.step()
        self.messenger.send("train_stop")

    def _two_party_val_per_epoch(self, testloader):
        self.bottom_model.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(testloader):
                smashed_data1 = self.bottom_model(X)
                self.messenger.send((smashed_data1.data, y))
        self.messenger.send("val_stop")


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "mnist"  # 'cifar10'
    model_name = "lenet5"  # 'VGG16'
    u_shape_flag = True
    server_ip, server_port = "localhost", 20000
    client_ip, client_port = "localhost", 25000
    _epochs = 20
    if u_shape_flag:
        _split_layer = [2, 4]
    else:
        _split_layer = [2]
    _batch_size = 64
    _learning_rate = 0.05
    _crypto_type = Const.PLAIN
    _loss_fn = nn.CrossEntropyLoss()

    # 1. Load datasets
    print("Loading dataset...")
    if dataset_name == "mnist":
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainset = datasets.MNIST(
            root="data", train=True, download=True, transform=transform_train
        )
        testset = datasets.MNIST(
            root="data", train=False, download=True, transform=transform_test
        )
    elif dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        trainset = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform_train
        )
        testset = datasets.CIFAR10(
            root="data", train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError("Invalid dataset name.")
    print("Done.")

    # 2. Create PyTorch model and optimizer
    if u_shape_flag:  # U-Shape split learning
        if model_name == "lenet5":
            client_model = SplitLenet5(_out_layer=_split_layer[0])
            top_model = SplitLenet5(_in_layer=_split_layer[1])
        elif model_name[0:3] == "VGG":
            client_model = SplitVGG(model_name, _out_layer=_split_layer[0])
            top_model = SplitVGG(model_name, _in_layer=_split_layer[1])
        else:
            raise ValueError("Invalid model name.")
        client_opt = torch.optim.SGD(client_model.parameters(), lr=_learning_rate)
        top_opt = torch.optim.SGD(top_model.parameters(), lr=_learning_rate)
    else:  # Two-party split learning
        if model_name == "lenet5":
            client_model, top_model = SplitLenet5(_out_layer=_split_layer[0]), None
        elif model_name[0:3] == "VGG":
            client_model, top_model = (
                SplitVGG(model_name, _out_layer=_split_layer[0]),
                None,
            )
        else:
            raise ValueError("Invalid model name.")
        client_opt, top_opt = (
            torch.optim.SGD(client_model.parameters(), lr=_learning_rate),
            None,
        )

    # 3. Initialize messenger
    _messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=server_ip,
        active_port=server_port,
        passive_ip=client_ip,
        passive_port=client_port,
    )

    # 4. Initilize NN protocol and start training
    client_party = ClientNN(
        epochs=_epochs,
        batch_size=_batch_size,
        bottom_model=client_model,
        bottom_opt=client_opt,
        messenger=_messenger,
        crypto_type=_crypto_type,
        u_shape_flag=u_shape_flag,
        top_model=top_model,
        top_opt=top_opt,
        loss_fn=_loss_fn,
    )
    client_party.train(trainset, testset)

    # 5. Close messenger, finish training
    _messenger.close()
