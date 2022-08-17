import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from termcolor import colored
import torch
from torch import nn

from linkefl.common.const import Const
from linkefl.common.factory import messenger_factory, crypto_factory
from linkefl.splitnn.model import *


class ServerNN:
    def __init__(self, 
                 epochs, 
                 batch_size, 
                 model,
                 optimizer, 
                 loss_fn, 
                 messenger, 
                 crypto_type, 
                 *,
                 precision=0.001, 
                 random_state=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.messenger = messenger
        self.crypto_type = crypto_type
        self.precision = precision
        self.random_state = random_state

    def train(self):
        self.model.train()
        start_time = None
        best_acc, best_auc = 0, 0
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch + 1))
            while True:
                recv_msg = self.messenger.recv()
                if recv_msg == 'train_stop': break
                (smashed_data, y) = recv_msg
                smashed_data.requires_grad_()
                if start_time is None: start_time = time.time()
                self.optimizer.zero_grad()
                outputs = self.model(smashed_data)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()
                self.messenger.send(smashed_data.grad)

            is_best = False
            curr_acc, curr_auc = self.validate()
            if curr_acc > best_acc:
                print(colored('Best model update.\n', 'red'))
                is_best = True
                best_acc = curr_acc
            if curr_auc > best_auc:
                best_auc = curr_auc
            self.messenger.send(is_best)

        print(colored('Total training and validation time: {:.4f}'.format(time.time() - start_time), 'red'))
        print(colored('Best testing accuracy: {:.5f}'.format(best_acc), 'red'))
        print(colored('Best testing auc: {:.5f}'.format(best_auc), 'red'))

    def validate(self):
        num_batches, num_samples, test_loss, correct = 0, 0, 0, 0
        labels, probs = np.array([]), np.array([]) # used for computing AUC score
        with torch.no_grad():
            while True:
                recv_msg = self.messenger.recv()
                if recv_msg == 'val_stop': break
                (smashed_data, y) = recv_msg
                outputs = self.model(smashed_data)
                labels = np.append(labels, y.numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(outputs[:, 1]).numpy())
                test_loss += self.loss_fn(outputs, y).item()
                correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
                num_batches += 1
                num_samples += y.size(0)
            test_loss /= num_batches
            acc = correct / num_samples
            n_classes = len(np.unique(labels))
            if n_classes == 2: auc = roc_auc_score(labels, probs)
            else: auc = 0
            print(f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%," f" Auc: {(100 * auc):>0.2f}%," f" Avg loss: {test_loss:>8f}")

            return acc, auc


if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'mnist'
    model_name = 'lenet5'
    server_ip, server_port = 'localhost', 20000
    client_ip, client_port = 'localhost', 25000
    _epochs = 20
    _split_layer = 2
    _batch_size = 64
    _learning_rate = 0.05
    _crypto_type = Const.PLAIN
    _loss_fn = nn.CrossEntropyLoss()

    # 2. Created PyTorch models and associated optimizers
    if model_name == 'lenet5':
        server_model = ServerLenet5(_split_layer)
    elif model_name[0:3] == 'VGG':
        server_model = ServerVGG(model_name, _split_layer)
    else:
        raise ValueError('Invalid dataset name.')
    _optimizer = torch.optim.SGD(server_model.parameters(), lr=_learning_rate)

    # 3. Initialize messenger
    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.ACTIVE_NAME,
                                   active_ip=server_ip,
                                   active_port=server_port,
                                   passive_ip=client_ip,
                                   passive_port=client_port)
    print('Server started, listening...')

    # 4. Initialize NN protocol and start training
    server_party = ServerNN(epochs=_epochs,
                             batch_size=_batch_size,
                             model=server_model,
                             optimizer=_optimizer,
                             loss_fn=_loss_fn,
                             messenger=_messenger,
                             crypto_type=_crypto_type)
    server_party.train()
    # 5. Close messenger, finish training
    _messenger.close()


