import datetime
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from termcolor import colored
import torch
from torch import nn
from torch.utils.data import DataLoader

from linkefl.common.const import Const
from linkefl.common.factory import messenger_factory, partial_crypto_factory
from linkefl.dataio import TorchDataset
from linkefl.util import num_input_nodes
from linkefl.vfl.nn.enc_layer import ActiveEncLayer
from linkefl.vfl.nn.model import MLPModel, CutLayer


class ActiveNeuralNetwork:
    def __init__(self,
                 epochs : int,
                 batch_size : int,
                 learning_rate : float,
                 models : dict,
                 optimizers : dict,
                 loss_fn,
                 messenger,
                 crypto_type,
                 *,
                 passive_in_nodes=None,
                 precision=0.001,
                 random_state=None,
                 saving_model=False,
                 model_path='./models'
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = models
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.messenger = messenger
        self.crypto_type = crypto_type
        self.passive_in_nodes = passive_in_nodes
        self.precision = precision
        self.random_state = random_state
        if random_state is not None:
            torch.random.manual_seed(random_state)
        self.saving_model = saving_model
        self.model_path = model_path
        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.ACTIVE_NAME,
            model_type=Const.VERTICAL_NN
        )

    def _sync_pubkey(self):
        print('Waiting for public key...')
        public_key = self.messenger.recv()
        print('Done!')
        return public_key

    def _init_dataloader(self, dataset, shuffle=False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self, trainset : TorchDataset, testset : TorchDataset):
        assert isinstance(trainset, TorchDataset), \
            'trainset should be an instance of TorchDataset'
        assert isinstance(testset, TorchDataset), \
            'testset should be an instance of TorchDataset'
        train_dataloader = self._init_dataloader(trainset)
        test_dataloader = self._init_dataloader(testset)

        public_key = self._sync_pubkey()
        self.cryptosystem = partial_crypto_factory(self.crypto_type, public_key=public_key)
        if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
            if self.passive_in_nodes is None:
                raise ValueError("when using encrypted NN protocol, you should specify "
                                 "the passive_in_nodes argument.")
            self.enc_layer = ActiveEncLayer(
                in_nodes=self.passive_in_nodes,
                out_nodes=self.models["cut"].out_nodes,
                eta=self.learning_rate,
                messenger=self.messenger,
                cryptosystem=self.cryptosystem,
                random_state=self.random_state,
                precision=self.precision
            )

        for model in self.models.values():
            model.train()

        start_time = time.time()
        best_acc, best_auc = 0, 0
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch))
            for batch_idx, (X, y) in enumerate(train_dataloader):
                # print(f"batch: {batch_idx}")
                # 1. forward
                active_repr = self.models["cut"](self.models["bottom"](X))
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    self.enc_layer.fed_forward()
                    passive_repr = self.messenger.recv()
                else:
                    passive_repr = self.messenger.recv()
                top_input = active_repr.data + passive_repr
                top_input = top_input.requires_grad_()
                logits = self.models["top"](top_input)
                loss = self.loss_fn(logits, y)

                # 2. backward
                for optmizer in self.optimizers.values():
                    optmizer.zero_grad()
                loss.backward()
                # update active party's top model, cut layer and bottom model
                active_repr.backward(top_input.grad)
                for optmizer in self.optimizers.values():
                    optmizer.step()
                # send back passive party's gradient
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    passive_grad = self.enc_layer.fed_backward(top_input.grad)
                    self.messenger.send(passive_grad)
                else:
                    self.messenger.send(top_input.grad)

                if batch_idx % 100 == 0:
                    loss, current = loss.item(), batch_idx * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{trainset.n_samples:>5d}]")

                # if batch_idx == 1:
                #     break

            is_best = False
            scores = self.validate(testset, existing_loader=test_dataloader)
            curr_acc, curr_auc = scores['acc'], scores['auc']
            if curr_auc == 0: # multi-class
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    is_best = True
                    print(colored('Best model updated.', 'red'))
                # no need to update best_auc here, because it always equals zero.
            else: # binary-class
                if curr_auc > best_auc:
                    best_auc = curr_auc
                    is_best = True
                    print(colored('Best model updated.', 'red'))
                if curr_acc > best_acc:
                    best_acc = curr_acc
            self.messenger.send(is_best)

        # close pool
        if hasattr(self, 'enc_layer'):
            self.enc_layer.close_pool()

        print(colored('Total training and validation time: {:.4f}'
                      .format(time.time() - start_time), 'red'))
        print(colored('Best testing accuracy: {:.5f}'.format(best_acc), 'red'))
        print(colored('Best testing auc: {:.5f}'.format(best_auc), 'red'))

    def validate(self, testset, existing_loader=None):
        if existing_loader is None:
            dataloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        n_batches = len(dataloader)
        test_loss, correct = 0.0, 0
        labels, probs = np.array([]), np.array([])
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                active_repr = self.models["cut"](self.models["bottom"](X))
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    self.enc_layer.fed_forward()
                    passive_repr = self.messenger.recv()
                else:
                    passive_repr = self.messenger.recv()
                top_input = active_repr + passive_repr
                logits = self.models["top"](top_input)
                labels = np.append(labels, y.numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(logits[:, 1]).numpy())
                test_loss += self.loss_fn(logits, y).item()
                correct += (logits.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= n_batches
            acc = correct / testset.n_samples
            n_classes = len(torch.unique(testset.labels))
            if n_classes == 2:
                auc = roc_auc_score(labels, probs)
            else:
                auc = 0
            print(f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%,"
                  f" Auc: {(100 * auc):>0.2f}%,"
                  f" Avg loss: {test_loss:>8f}")

            scores = {"acc": acc, "auc": auc, "loss": test_loss}
            self.messenger.send(scores)
            return scores

    def train_alone(self, trainset: TorchDataset, testset: TorchDataset):
        assert isinstance(trainset, TorchDataset), \
            'trainset should be an instance of TorchDataset'
        assert isinstance(testset, TorchDataset), \
            'testset should be an instance of TorchDataset'
        train_dataloader = self._init_dataloader(trainset)
        test_dataloader = self._init_dataloader(testset)

        for model in self.models.values():
            model.train()

        start_time = time.time()
        best_acc, best_auc = 0, 0
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch))
            for batch_idx, (X, y) in enumerate(train_dataloader):
                # forward
                logits = self.models["top"](
                    self.models["cut"](
                        self.models["bottom"](X)
                    )
                )
                loss = self.loss_fn(logits, y)

                # backward
                for optmizer in self.optimizers.values():
                    optmizer.zero_grad()
                loss.backward()
                for optmizer in self.optimizers.values():
                    optmizer.step()
                if batch_idx % 100 == 0:
                    loss, current = loss.item(), batch_idx * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{trainset.n_samples:>5d}]")

            scores = self.validate_alone(testset, existing_loader=test_dataloader)
            curr_acc, curr_auc = scores['acc'], scores['auc']
            if curr_auc == 0:  # multi-class
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    print(colored('Best model updated.', 'red'))
                # no need to update best_auc here, because it always equals zero.
            else:  # binary-class
                if curr_auc > best_auc:
                    best_auc = curr_auc
                    print(colored('Best model updated.', 'red'))
                if curr_acc > best_acc:
                    best_acc = curr_acc

        print(colored('Total training and validation time: {:.4f}'
                      .format(time.time() - start_time), 'red'))
        print(colored('Best testing accuracy: {:.5f}'.format(best_acc), 'red'))
        print(colored('Best testing auc: {:.5f}'.format(best_auc), 'red'))

    def validate_alone(self, testset: TorchDataset, existing_loader=None):
        if existing_loader is None:
            dataloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        n_batches = len(dataloader)
        test_loss, correct = 0.0, 0
        labels, probs = np.array([]), np.array([])
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                logits = self.models["top"](
                    self.models["cut"](
                        self.models["bottom"](X)
                    )
                )
                labels = np.append(labels, y.numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(logits[:, 1]).numpy())
                test_loss += self.loss_fn(logits, y).item()
                correct += (logits.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= n_batches
            acc = correct / testset.n_samples
            n_classes = len(torch.unique(testset.labels))
            if n_classes == 2:
                auc = roc_auc_score(labels, probs)
            else:
                auc = 0
            print(f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%,"
                  f" Auc: {(100 * auc):>0.2f}%,"
                  f" Avg loss: {test_loss:>8f}")

            scores = {"acc": acc, "auc": auc, "loss": test_loss}
            return scores


if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'mnist'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 30000
    _epochs = 100
    _batch_size = 100
    _learning_rate = 0.01
    _passive_in_nodes = 128
    _crypto_type = Const.PLAIN
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = 1314

    # 1. Load datasets
    print('Loading dataset...')
    active_trainset = TorchDataset.buildin_dataset(dataset_name=dataset_name,
                                                   role=Const.ACTIVE_NAME,
                                                   root='../data',
                                                   train=True,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option,
                                                   seed=_random_state)
    active_testset = TorchDataset.buildin_dataset(dataset_name=dataset_name,
                                                  role=Const.ACTIVE_NAME,
                                                  root='../data',
                                                  train=False,
                                                  download=True,
                                                  passive_feat_frac=passive_feat_frac,
                                                  feat_perm_option=feat_perm_option,
                                                  seed=_random_state)
    print('Done.')

    # 2. Create PyTorch models and optimizers
    input_nodes = num_input_nodes(
        dataset_name=dataset_name,
        role=Const.ACTIVE_NAME,
        passive_feat_frac=passive_feat_frac
    )
    # mnist & fashion_mnist
    bottom_nodes = [input_nodes, 256, 128]
    cut_nodes = [128, 64]
    top_nodes = [64, 10]

    # criteo
    # bottom_nodes = [input_nodes, 15, 10]
    # cut_nodes = [10, 10]
    # top_nodes = [10, 2]

    # census
    # bottom_nodes = [input_nodes, 20, 10]
    # cut_nodes = [10, 8]
    # top_nodes = [8, 2]

    # epsilon
    # bottom_nodes = [input_nodes, 25, 10]
    # cut_nodes = [10, 10]
    # top_nodes = [10, 2]
    bottom_model = MLPModel(bottom_nodes,
                            activate_input=False,
                            activate_output=True,
                            random_state=_random_state)
    cut_layer = CutLayer(*cut_nodes, random_state=_random_state)
    top_model = MLPModel(top_nodes,
                         activate_input=True,
                         activate_output=False,
                         random_state=_random_state)
    _models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}
    _optimizers = {name: torch.optim.SGD(model.parameters(), lr=_learning_rate)
                   for name, model in _models.items()}

    # 3. Initialize messenger
    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.ACTIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)
    print('Active party started, listening...')

    # 4. Initialize vertical NN protocol and start training
    active_party = ActiveNeuralNetwork(epochs=_epochs,
                                       batch_size=_batch_size,
                                       learning_rate=_learning_rate,
                                       models=_models,
                                       optimizers=_optimizers,
                                       loss_fn=_loss_fn,
                                       messenger=_messenger,
                                       crypto_type=_crypto_type,
                                       passive_in_nodes=_passive_in_nodes,
                                       random_state=_random_state)
    active_party.train(active_trainset, active_testset)

    # 5. Close messenger, finish training
    _messenger.close()