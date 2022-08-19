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
                 u_shape_flag,
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
        self.u_shape_flag = u_shape_flag
        self.precision = precision
        self.random_state = random_state
        self.start_time = None

    def train(self):
        if self.u_shape_flag == True: 
            self._u_shape_train()
        else: 
            self._two_party_train()

    def _u_shape_train(self):
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch + 1))
            self._u_shape_train_per_epoch()
            self._u_shape_val_per_epoch()
            is_best = self.messenger.recv()
            if is_best: 
                print(colored('Best model updated.', 'red'))
            print(f"Epoch {epoch + 1} finished.\n")
        print(colored('Total training and validation time: {:.2f}'.format(time.time() - self.start_time), 'red'))

    def _two_party_train(self):
        best_acc, best_auc = 0, 0
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch + 1))
            self._two_party_train_per_epoch()
            curr_acc, curr_auc = self._two_party_val_per_epoch()
            is_best = False
            if curr_acc > best_acc:
                print(colored('Best model update.\n', 'red'))
                is_best, best_acc = True, curr_acc
            if curr_auc > best_auc: 
                best_auc = curr_auc
            self.messenger.send(is_best)
        print(colored('Total training and validation time: {:.2f}'.format(time.time() - self.start_time), 'red'))
        print(colored('Best testing accuracy: {:.5f}'.format(best_acc), 'red'))
        print(colored('Best testing auc: {:.5f}'.format(best_auc), 'red'))

    def _u_shape_train_per_epoch(self):
        self.model.train()
        while True:
            recv_msg = self.messenger.recv()
            if recv_msg == 'train_stop': 
                break
            smashed_data1 = recv_msg
            smashed_data1.requires_grad_()
            if self.start_time is None: 
                self.start_time = time.time()
            self.optimizer.zero_grad()
            smashed_data2 = self.model(smashed_data1)
            self.messenger.send(smashed_data2.data)

            grads = self.messenger.recv()
            smashed_data2.backward(grads)
            self.optimizer.step()
            self.messenger.send(smashed_data1.grad)

    def _u_shape_val_per_epoch(self):
        self.model.eval()
        while True:
            recv_msg = self.messenger.recv()
            if recv_msg == 'val_stop': 
                break
            with torch.no_grad():
                smashed_data1 = recv_msg
                smashed_data2 = self.model(smashed_data1)
                self.messenger.send(smashed_data2.data)
    
    def _two_party_train_per_epoch(self):
        self.model.train()
        while True:
            recv_msg = self.messenger.recv()
            if recv_msg == 'train_stop': 
                break
            (smashed_data1, y) = recv_msg
            smashed_data1.requires_grad_()
            if self.start_time is None: 
                self.start_time = time.time()
            self.optimizer.zero_grad()
            preds = self.model(smashed_data1)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()
            self.messenger.send(smashed_data1.grad)

    def _two_party_val_per_epoch(self):
        self.model.eval()
        num_batches, num_samples, test_loss, correct = 0, 0, 0, 0
        labels, probs = np.array([]), np.array([]) # used for computing AUC score
        while True:
            recv_msg = self.messenger.recv()
            if recv_msg == 'val_stop': 
                break
            with torch.no_grad():
                (smashed_data1, y) = recv_msg
                preds = self.model(smashed_data1)
                labels = np.append(labels, y.numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(preds[:, 1]).numpy())
                test_loss += self.loss_fn(preds, y).item()
                correct += (preds.argmax(1) == y).type(torch.float).sum().item()
                num_batches += 1
                num_samples += y.size(0)
        test_loss /= num_batches
        acc = correct / num_samples
        n_classes = len(np.unique(labels))
        if n_classes == 2: 
            auc = roc_auc_score(labels, probs)
        else: 
            auc = 0
        print(f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%," f" Auc: {(100 * auc):>0.2f}%," f" Avg loss: {test_loss:>8f}")
        return acc, auc

if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'mnist'
    model_name = 'lenet5'
    u_shape_flag = True
    server_ip, server_port = 'localhost', 20000
    client_ip, client_port = 'localhost', 25000
    _epochs = 20
    if u_shape_flag == True: 
        _split_layer = [2, 4]
    else: 
        _split_layer = [2]
    _batch_size = 64
    _learning_rate = 0.05
    _crypto_type = Const.PLAIN
    _loss_fn = nn.CrossEntropyLoss()

    # 2. Created PyTorch models and associated optimizers
    if model_name == 'lenet5':
        if u_shape_flag == False: 
            server_model = SplitLenet5(_in_layer=_split_layer[0])
        else: 
            server_model = SplitLenet5(_in_layer=_split_layer[0],
                                       _out_layer=_split_layer[1])
    elif model_name[0:3] == 'VGG':
        if u_shape_flag == False: 
            server_model = SplitVGG(model_name, _in_layer=_split_layer[0])
        else: 
            server_model = SplitVGG(model_name, _in_layer=_split_layer[0], 
                                    _out_layer=_split_layer[1])
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
                             crypto_type=_crypto_type,
                             u_shape_flag=u_shape_flag)
    server_party.train()

    # 5. Close messenger, finish training
    _messenger.close()
