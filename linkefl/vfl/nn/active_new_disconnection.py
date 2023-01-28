import datetime
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from termcolor import colored
from torch import nn
from torch.utils.data import DataLoader

from linkefl.common.const import Const
from linkefl.common.factory import (
    messenger_factory,
    messenger_factory_multi_disconnection,
    partial_crypto_factory,
)
from linkefl.dataio import TorchDataset
from linkefl.util import num_input_nodes
from linkefl.vfl.nn.enc_layer import ActiveEncLayer
from linkefl.vfl.nn.model import CutLayer, MLPModel


class ActiveNeuralNetwork_disconnection:
    def __init__(
        self,
        epochs,
        batch_size,
        learning_rate,
        models,
        optimizers,
        loss_fn,
        messenger,
        crypto_type,
        *,
        world_size=1,
        passive_in_nodes=None,
        precision=0.001,
        random_state=None,
        saving_model=False,
        model_path="./models",
        reconnection=False,
        reconnect_port=["30001"],
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = models
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.messenger = messenger
        self.world_size = 1
        self.reconnection = reconnection
        self.reconnect_port = reconnect_port
        public_key = self._sync_pubkey()
        self.crypto_type = crypto_type
        self.cryptosystem = partial_crypto_factory(crypto_type, public_key=public_key)
        if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
            if passive_in_nodes is None:
                raise ValueError(
                    "when using encrypted NN protocol, you should specify "
                    "the passive_in_nodes argument."
                )
            self.enc_layer = ActiveEncLayer(
                in_nodes=passive_in_nodes,
                out_nodes=self.models["cut"].out_nodes,
                eta=self.learning_rate,
                messenger=self.messenger,
                cryptosystem=self.cryptosystem,
                precision=precision,
            )

        self.precision = precision
        self.random_state = random_state
        self.saving_model = saving_model
        self.model_path = model_path
        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.ACTIVE_NAME,
            model_type=Const.VERTICAL_NN,
        )

    def _sync_pubkey(self):
        print("Waiting for public key...")
        # 暂时只支持两方
        public_key = self.messenger.recv()
        print("Done!")
        return public_key

    def _init_dataloader(self, dataset, shuffle=False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def get_init_output(self, passive_party):
        init_output, passive_party = self.messenger.recv(passive_party=passive_party)
        return init_output, passive_party

    def try_to_connection(self, reconnect_port):
        print(colored("Try To Connection", "red"))
        passive_party = self.messenger.try_reconnect(reconnect_port)
        if passive_party:
            public_key = self._sync_pubkey()
            self.cryptosystem = partial_crypto_factory(
                self.crypto_type, public_key=public_key
            )
        if passive_party:
            print(colored("Reconnection Successfully!!!", "red"))
        else:
            print(colored("Reconnection Failed!!!", "red"))

        return passive_party

    def train(self, trainset, testset):
        assert isinstance(
            trainset, TorchDataset
        ), "trainset should be an instance of TorchDataset"
        assert isinstance(
            testset, TorchDataset
        ), "testset should be an instance of TorchDataset"
        train_dataloader = self._init_dataloader(trainset)
        test_dataloader = self._init_dataloader(testset)

        for model in self.models.values():
            model.train()
        print("begin train...")

        passive_party = True

        init_output_train, passive_party = self.get_init_output(
            passive_party=passive_party
        )
        init_output_test, passive_party = self.get_init_output(
            passive_party=passive_party
        )

        if passive_party:
            print("[ACTIVE] Init Done!")
        else:
            print(colored("Initialization failed, please restart training!!!", "red"))
            return 0

        start_time = time.time()
        best_acc, best_auc = 0, 0
        out_temp = init_output_train
        output_temp_test = init_output_test

        for epoch in range(self.epochs):
            if self.reconnection and passive_party is False:
                passive_party = self.try_to_connection(self.reconnect_port)
                self.messenger.send(epoch, passive_party=passive_party)

            print("Epoch: {}".format(epoch + 1))
            if passive_party is True:
                print(colored("passive_party status: connection", "red"))
            else:
                print(colored("passive_party status: disconnection", "red"))

            for batch_idx, (X, y) in enumerate(train_dataloader):
                # 1.forward
                active_repr = self.models["cut"](self.models["bottom"](X))

                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    self.enc_layer.fed_forward()
                    passive_repr, passive_party = self.messenger.recv(
                        passive_party=passive_party
                    )
                else:
                    passive_repr, passive_party = self.messenger.recv(
                        passive_party=passive_party
                    )

                if passive_party:
                    passive_data = passive_repr
                    out_temp[batch_idx] = passive_data
                else:
                    passive_data = out_temp[batch_idx]

                top_input = active_repr.data + passive_data

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
                    passive_party = self.messenger.send(
                        passive_grad, passive_party=passive_party
                    )
                else:
                    passive_party = self.messenger.send(
                        top_input.grad, passive_party=passive_party
                    )

                if batch_idx % 100 == 0:
                    loss, current = loss.item(), batch_idx * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{trainset.n_samples:>5d}]")
                    # break

            is_best = False
            scores, passive_party, output_temp_test = self.validate(
                testset,
                existing_loader=test_dataloader,
                output_temp_test=output_temp_test,
                passive_party=passive_party,
            )
            curr_acc, curr_auc = scores["acc"], scores["auc"]
            if curr_auc == 0:  # multi-class
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    is_best = True
                    print(colored("Best model updated.", "red"))
                # no need to update best_auc here, because it always equals zero.
            else:  # binary-class
                if curr_auc > best_auc:
                    best_auc = curr_auc
                    is_best = True
                    print(colored("Best model updated.", "red"))
                if curr_acc > best_acc:
                    best_acc = curr_acc
            passive_party = self.messenger.send(is_best, passive_party=passive_party)

        print(
            colored(
                "Total training and validation time: {:.4f}".format(
                    time.time() - start_time
                ),
                "red",
            )
        )
        print(colored("Best testing accuracy: {:.5f}".format(best_acc), "red"))
        print(colored("Best testing auc: {:.5f}".format(best_auc), "red"))

    def validate(self, testset, output_temp_test, passive_party, existing_loader=None):
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
                    passive_repr, passive_party = self.messenger.recv(
                        passive_party=passive_party
                    )
                else:
                    passive_repr, passive_party = self.messenger.recv(
                        passive_party=passive_party
                    )

                if passive_party:
                    passive_data = passive_repr
                    output_temp_test[batch] = passive_data.data
                else:
                    passive_data = output_temp_test[batch]

                top_input = active_repr + passive_data
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
            print(
                f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%,"
                f" Auc: {(100 * auc):>0.2f}%,"
                f" Avg loss: {test_loss:>8f}"
            )

            scores = {"acc": acc, "auc": auc, "loss": test_loss}
            passive_party = self.messenger.send(scores, passive_party=passive_party)
            return scores, passive_party, output_temp_test


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "census"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = "localhost"
    active_port = 20000
    passive_ip = ["localhost"]
    passive_port = [30000]
    reconnect_port = [30001]
    _epochs = 100
    _batch_size = 100
    _learning_rate = 0.01
    _passive_in_nodes = 10
    _crypto_type = Const.PLAIN
    _loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(1314)
    world_size = 1
    reconnection = True
    # 1. Load datasets
    print("Loading dataset...")
    active_trainset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.ACTIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_testset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.ACTIVE_NAME,
        root="../data",
        train=False,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    print("Done.")

    # 2. Create PyTorch models and optimizers
    input_nodes = num_input_nodes(
        dataset_name=dataset_name,
        role=Const.ACTIVE_NAME,
        passive_feat_frac=passive_feat_frac,
    )
    # mnist & fashion_mnist
    # bottom_nodes = [input_nodes, 256, 128]
    # cut_nodes = [128, 64]
    # top_nodes = [64, 10]

    # criteo
    # bottom_nodes = [input_nodes, 15, 10]
    # cut_nodes = [10, 10]
    # top_nodes = [10, 2]

    # census
    bottom_nodes = [input_nodes, 20, 10]
    cut_nodes = [10, 8]
    top_nodes = [8, 2]
    bottom_model = MLPModel(bottom_nodes, activate_input=False, activate_output=True)
    cut_layer = CutLayer(*cut_nodes)
    top_model = MLPModel(top_nodes, activate_input=True, activate_output=False)
    _models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}
    _optimizers = {
        name: torch.optim.SGD(model.parameters(), lr=_learning_rate)
        for name, model in _models.items()
    }

    # 3. Initialize messenger
    _messenger = messenger_factory_multi_disconnection(
        messenger_type=Const.FAST_SOCKET,
        role=Const.ACTIVE_NAME,
        model_type="NN",
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )
    print("Active party started, listening...")

    # 4. Initialize vertical NN protocol and start training
    active_party = ActiveNeuralNetwork_disconnection(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=_models,
        optimizers=_optimizers,
        loss_fn=_loss_fn,
        messenger=_messenger,
        crypto_type=_crypto_type,
        passive_in_nodes=_passive_in_nodes,
        reconnection=reconnection,
        reconnect_port=reconnect_port,
    )
    active_party.train(active_trainset, active_testset)

    # 5. Close messenger, finish training
    _messenger.close()
