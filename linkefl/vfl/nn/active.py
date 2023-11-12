import datetime
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from termcolor import colored
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from linkefl.base import BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.factory import logger_factory, partial_crypto_factory
from linkefl.common.log import GlobalLogger
from linkefl.dataio import MediaDataset, TorchDataset  # noqa: F403
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *  # noqa: F403
from linkefl.util.progress import progress_bar
from linkefl.vfl.nn.enc_layer import ActiveEncLayer
from linkefl.vfl.utils.evaluate import Plot


def loss_reweight(y):
    if type(y) == torch.Tensor:
        y = y.numpy()
    elif type(y) == np.ndarray:
        pass
    else:
        raise ValueError("Only tensor and ndarray are supported!")

    unique_labels, count = np.unique(y, return_counts=True)

    weight = 1 / count
    weight /= np.sum(weight)
    weight *= len(unique_labels)  # Normalization for numerical stability

    return torch.FloatTensor(weight)


class ActiveNeuralNetwork(BaseModelComponent):
    def __init__(
        self,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Optimizer],
        loss_fn: _Loss,
        messengers: List[BaseMessenger],
        logger: GlobalLogger,
        schedulers: Optional[Dict[str, Any]] = None,
        topk: int = 1,
        rank: int = 0,
        num_workers: int = 1,
        val_freq: int = 1,
        device: str = "cpu",
        encode_precision: float = 0.001,
        random_state: Optional[int] = None,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        save_every_epoch=False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = models
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.messengers = messengers
        self.logger = logger
        self.schedulers = schedulers
        self.topk = topk
        self.rank = rank
        self.num_workers = num_workers
        self.val_freq = val_freq
        self.device = device
        self.encode_precision = encode_precision
        self.random_state = random_state
        self.save_every_epoch = save_every_epoch
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic = True  # important
            torch.backends.cudnn.benchmark = False
        self.saving_model = saving_model
        if self.saving_model:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if model_dir is None:
                default_dir = "models"
                model_dir = os.path.join(default_dir, self.create_time)
            if model_name is None:
                model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.ACTIVE_NAME,
                        algo_name=Const.AlgoNames.VFL_NN,
                    )
                    + ".model"
                )
            self.model_dir = model_dir
            self.model_name = model_name
            self.pics_dir = self.model_dir
            if not os.path.exists(self.model_dir):
                pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # exists when training started
        self.cryptosystem_list: list = []
        self.enc_layer_list: list = []

    def _sync_pubkey(self):
        print("Waiting for public key...")
        public_key_list, crypto_type_list = [], []
        for msger in self.messengers:
            pubkey, crypto_type = msger.recv()
            public_key_list.append(pubkey)
            crypto_type_list.append(crypto_type)
        print("Done!")
        return public_key_list, crypto_type_list

    def _init_dataloader(
        self, dataset, shuffle=False, num_workers=1, persistent_workers=True, bs=None,
    ):
        if bs is None:
            bs = dataset.n_samples if self.batch_size == -1 else self.batch_size
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        return dataloader

    def train(self, trainset: TorchDataset, testset: TorchDataset) -> None:
        assert isinstance(
            trainset, TorchDataset
        ), "trainset should be an instance of TorchDataset"
        assert isinstance(
            testset, TorchDataset
        ), "testset should be an instance of TorchDataset"
        train_dataloader = self._init_dataloader(trainset, shuffle=True, num_workers=2)
        test_dataloader = self._init_dataloader(testset, num_workers=2)

        public_key_list, crypto_type_list = self._sync_pubkey()
        for idx, (public_key, crypto_type) in enumerate(
            zip(public_key_list, crypto_type_list)
        ):
            cryptosystem = partial_crypto_factory(crypto_type, public_key=public_key)
            if crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
                in_nodes = self.messengers[idx].recv()
                enc_layer = ActiveEncLayer(
                    in_nodes=in_nodes,
                    out_nodes=self.models["cut"].out_nodes,
                    eta=self.learning_rate,
                    messenger=self.messengers[idx],
                    cryptosystem=cryptosystem,
                    num_workers=self.num_workers,
                    random_state=self.random_state,
                    encode_precision=self.encode_precision,
                )
            else:
                enc_layer = None
            self.cryptosystem_list.append(cryptosystem)
            self.enc_layer_list.append(enc_layer)

        start_time = time.time()
        best_acc, best_auc = 0.0, 0.0
        if self.topk > 1:
            best_topk_acc = 0.0
        train_acc_records, valid_acc_records = [], []
        train_auc_records, valid_auc_records = [], []
        train_loss_records, valid_loss_records = [], []
        for epoch in range(self.epochs):
            train_loss, correct, total = 0, 0, 0
            if self.topk > 1:
                topk_correct = 0
            for model in self.models.values():
                model.train()
            self.logger.log(f"Epoch {epoch} started...")
            print(f"Epoch: {epoch}")
            torch.manual_seed(epoch)  # fix dataloader batching order when shuffle=True
            for batch_idx, (X, y) in enumerate(train_dataloader):
                # 1. forward
                X = X.to(self.device)
                y = y.to(self.device)
                active_repr = self.models["cut"](self.models["bottom"](X))
                top_input = active_repr.data
                for idx, msger in enumerate(self.messengers):
                    if self.cryptosystem_list[idx].type in (
                        Const.PAILLIER,
                        Const.FAST_PAILLIER,
                    ):
                        self.enc_layer_list[idx].fed_forward()
                        passive_repr = msger.recv().to(self.device)
                    else:
                        passive_repr = msger.recv().to(self.device)
                    top_input += passive_repr
                # top_input = active_repr.data + passive_repr
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
                for idx, msger in enumerate(self.messengers):
                    if self.cryptosystem_list[idx].type in (
                        Const.PAILLIER,
                        Const.FAST_PAILLIER,
                    ):
                        passive_grad = self.enc_layer_list[idx].fed_backward(
                            top_input.grad
                        )
                        msger.send(passive_grad)
                    else:
                        msger.send(top_input.grad)

                train_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
                progress_bar(
                    batch_idx,
                    len(train_dataloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
                if self.topk > 1:
                    _, predicted = logits.topk(self.topk, dim=1)
                    y = y.view(y.size(0), -1).expand_as(predicted)
                    topk_correct += predicted.eq(y).sum().item()

            # update learning rate scheduler
            if self.schedulers is not None:
                for scheduler in self.schedulers.values():
                    scheduler.step()

            # validate model
            is_best = False
            if (epoch + 1) % self.val_freq == 0:
                scores = self.validate(testset, existing_loader=test_dataloader)
                train_loss_records.append(train_loss / len(train_dataloader))
                valid_loss_records.append(scores["loss"])
                train_auc_records.append(0)  # TODO
                valid_auc_records.append(scores["auc"])
                train_acc_records.append(correct / total)
                valid_acc_records.append(scores["acc"])
                self.logger.log_metric(
                    epoch=epoch + 1,
                    loss=scores["loss"],
                    acc=scores["acc"],
                    auc=scores["auc"],
                    total_epoch=self.epochs,
                )
                if scores["auc"] == 0:  # multi-class
                    if scores["acc"] > best_acc:
                        best_acc = scores["acc"]
                        is_best = True
                    if self.topk > 1:
                        if scores["topk_acc"] > best_topk_acc:
                            best_topk_acc = scores["topk_acc"]
                            print("best topk_acc updated.")
                    # no need to update best_auc here, because it always equals zero.
                else:  # binary-class
                    if scores["auc"] > best_auc:
                        best_auc = scores["auc"]
                        is_best = True
                    if scores["acc"] > best_acc:
                        best_acc = scores["acc"]
                if is_best:
                    print(colored("Best model updated.", "red"))
                    self.logger.log("Best model updates.")
                    if self.saving_model:
                        TorchModelIO.save(
                            self.models,
                            self.model_dir,
                            self.model_name,
                            epoch=epoch,
                        )
                if self.save_every_epoch:
                    TorchModelIO.save(
                        self.models,
                        self.model_dir,
                        f"active_epoch_{epoch}.model",
                        epoch=epoch,
                    )
                for msger in self.messengers:
                    msger.send(is_best)

        # close pool
        for enc_layer in self.enc_layer_list:
            if enc_layer is not None:
                enc_layer.close_pool()

        if self.saving_model:
            Plot.plot_train_test_loss(
                train_loss_records, valid_loss_records, self.pics_dir
            )
            Plot.plot_train_test_acc(
                train_acc_records, valid_acc_records, self.pics_dir
            )
            Plot.plot_train_test_auc(
                train_auc_records, valid_auc_records, self.pics_dir
            )

        print(colored(f"elapsed time: {time.time() - start_time}", "red"))
        print(colored("Best testing accuracy: {:.5f}".format(best_acc), "red"))
        if self.topk > 1:
            print(colored("Best testing topK accuracy: {:.5f}".format(best_topk_acc), "red"))
        print(colored("Best testing auc: {:.5f}".format(best_auc), "red"))
        self.logger.log(f"elapsed time: {time.time() - start_time}")
        self.logger.log("Best testing accuracy: {:.5f}".format(best_acc))
        self.logger.log("Best testing auc: {:.5f}".format(best_auc))

    def validate(
        self,
        testset: TorchDataset,
        *,
        existing_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        if existing_loader is None:
            dataloader = self._init_dataloader(testset, num_workers=2)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        n_batches = len(dataloader)
        test_loss, correct, total = 0, 0, 0
        if self.topk > 1:
            topk_correct = 0
        labels, probs = np.array([]), np.array([])
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                active_repr = self.models["cut"](self.models["bottom"](X))
                top_input = active_repr
                for idx, msger in enumerate(self.messengers):
                    if self.cryptosystem_list[idx].type in (
                        Const.PAILLIER,
                        Const.FAST_PAILLIER,
                    ):
                        self.enc_layer_list[idx].fed_forward()
                        passive_repr = msger.recv()
                    else:
                        passive_repr = msger.recv().to(self.device)
                    top_input += passive_repr
                # top_input = active_repr + passive_repr
                logits = self.models["top"](top_input)
                labels = np.append(labels, y.cpu().numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(logits[:, 1]).cpu().numpy())
                test_loss += self.loss_fn(logits, y).item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
                progress_bar(
                    batch_idx,
                    len(dataloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
                if self.topk > 1:
                    _, predicted = logits.topk(self.topk, dim=1)
                    y = y.view(y.size(0), -1).expand_as(predicted)
                    topk_correct += predicted.eq(y).sum().item()

            test_loss /= n_batches
            acc = correct / total
            n_classes = len(torch.unique(testset.labels))
            if n_classes == 2:
                auc = roc_auc_score(labels, probs)
            else:
                auc = 0

            scores = {"acc": acc, "auc": auc, "loss": test_loss}
            if self.topk > 1:
                scores["topk_acc"] = topk_correct / total
            return scores

    def fit(
        self,
        trainset: TorchDataset,
        validset: TorchDataset,
        role: str = Const.ACTIVE_NAME,
    ) -> None:
        self.train(trainset, validset)

    def score(
        self, testset: TorchDataset, role: str = Const.ACTIVE_NAME
    ) -> Dict[str, float]:
        return self.validate(testset)

    def train_alone(self, trainset: TorchDataset, testset: TorchDataset) -> None:
        assert isinstance(
            trainset, TorchDataset
        ), "trainset should be an instance of TorchDataset"
        assert isinstance(
            testset, TorchDataset
        ), "testset should be an instance of TorchDataset"
        train_dataloader = self._init_dataloader(trainset, shuffle=True, num_workers=2, bs=self.batch_size)
        test_dataloader = self._init_dataloader(testset, num_workers=2, bs=256)

        start_time = time.time()
        best_acc, best_auc = 0, 0
        if self.topk > 1:
            best_topk_acc = 0
        for epoch in range(self.epochs):
            for model in self.models.values():
                model.train()
            correct, total = 0, 0
            train_loss = 0
            if self.topk > 1:
                topk_correct = 0
            print("Epoch: {}".format(epoch))
            for batch_idx, (X, y) in enumerate(train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                # forward
                logits = self.models["top"](
                    self.models["cut"](self.models["bottom"](X))
                )
                loss = self.loss_fn(logits, y)
                train_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
                progress_bar(
                    batch_idx,
                    len(train_dataloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
                if self.topk > 1:
                    _, predicted = logits.topk(self.topk, dim=1)
                    y = y.view(y.size(0), -1).expand_as(predicted)
                    topk_correct += predicted.eq(y).sum().item()

                # backward
                for optmizer in self.optimizers.values():
                    optmizer.zero_grad()
                loss.backward()
                for optmizer in self.optimizers.values():
                    optmizer.step()

            if self.schedulers is not None:
                for scheduler in self.schedulers.values():
                    scheduler.step()

            scores, _ = self.validate_alone(testset, existing_loader=test_dataloader)
            curr_acc, curr_auc = scores["acc"], scores["auc"]
            if curr_auc == 0:  # multi-class
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    print(colored("Best model updated.", "red"))
                    if self.saving_model:
                        TorchModelIO.save(
                            self.models,
                            self.model_dir,
                            self.model_name,
                            epoch=epoch,
                        )
                if self.topk > 1:
                    if scores["topk_acc"] > best_topk_acc:
                        best_topk_acc = scores["topk_acc"]
                        print("best topk_acc updated.")
                # no need to update best_auc here, because it always equals zero.
            else:  # binary-class
                if curr_auc > best_auc:
                    best_auc = curr_auc
                    print(colored("Best model updated.", "red"))
                if curr_acc > best_acc:
                    best_acc = curr_acc

        print(colored(f"elapsed time: {time.time() - start_time}", "red"))
        print(colored("Best testing accuracy: {:.5f}".format(best_acc), "red"))
        if self.topk > 1:
            print(colored("Best testing topK accuracy: {:.5f}".format(best_topk_acc), "red"))
        print(colored("Best testing auc: {:.5f}".format(best_auc), "red"))
        return best_acc

    def validate_alone(
        self,
        testset: TorchDataset,
        *,
        existing_loader: Optional[DataLoader] = None,
    ):
        if existing_loader is None:
            dataloader = self._init_dataloader(testset, num_workers=2, bs=256)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        total_embeddings = None
        start_idx = 0
        n_batches = len(dataloader)
        test_loss, correct, total = 0.0, 0, 0
        if self.topk > 1:
            topk_correct = 0
        labels, probs = np.array([]), np.array([])
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                embedding = self.models["bottom"](X)
                if total_embeddings is None:
                    total_embeddings = torch.zeros(len(testset), embedding.size(1)).to(self.device)
                index = torch.arange(start_idx, start_idx + X.size(0)).to(self.device)
                total_embeddings.index_copy_(0, index, embedding)
                start_idx = start_idx + X.size(0)

                logits = self.models["top"](self.models["cut"](embedding))
                labels = np.append(labels, y.cpu().numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(logits[:, 1]).cpu().numpy())
                test_loss += self.loss_fn(logits, y).item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
                progress_bar(
                    batch_idx,
                    len(dataloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
                if self.topk > 1:
                    _, predicted = logits.topk(self.topk, dim=1)
                    y = y.view(y.size(0), -1).expand_as(predicted)
                    topk_correct += predicted.eq(y).sum().item()

            test_loss /= n_batches
            acc = correct / total
            n_classes = len(torch.unique(testset.labels))
            if n_classes == 2:
                auc = roc_auc_score(labels, probs)
            else:
                auc = 0

            scores = {"acc": acc, "auc": auc, "loss": test_loss}
            if self.topk > 1:
                scores["topk_acc"] = topk_correct / total
            return scores, total_embeddings

    @staticmethod
    def online_inference(
        dataset: TorchDataset,
        messengers: List[BaseMessenger],
        logger: GlobalLogger,
        model_dir: str,
        model_name: str,
        role: str = Const.ACTIVE_NAME,
    ):
        models: dict = TorchModelIO.load(model_dir, model_name)["model"]
        for model in models.values():
            model.eval()
        dataloader = DataLoader(dataset, batch_size=dataset.n_samples, shuffle=False)

        # n_batches = len(dataloader)
        # test_loss, correct = 0.0, 0
        # labels, probs = np.array([]), np.array([])
        # loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            preds = None
            for batch, (X, y) in enumerate(dataloader):
                active_repr = models["cut"](models["bottom"](X))
                top_input = active_repr
                for msger in messengers:
                    passive_repr = msger.recv()
                    top_input += passive_repr
                logits = models["top"](top_input)
                # labels = np.append(labels, y.cpu().numpy().astype(np.int32))
                # probs = np.append(probs, torch.sigmoid(logits[:, 1]).cpu().numpy())
                # test_loss += loss_fn(logits, y).item()
                # correct += (logits.argmax(1) == y).type(torch.float).sum().item()
                if preds is None:
                    preds = logits.argmax(1)
                else:
                    preds = torch.concat((preds, logits.argmax(1)), dim=0)
            # test_loss /= n_batches
            # acc = correct / dataset.n_samples
            # n_classes = len(torch.unique(dataset.labels))
            # if n_classes == 2:
            #     auc = roc_auc_score(labels, probs)
            # else:
            #     auc = 0
            # print(
            #     f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%,"
            #     f" Auc: {(100 * auc):>0.2f}%,"
            #     f" Avg loss: {test_loss:>8f}"
            # )

            # scores = {"acc": acc, "auc": auc, "loss": test_loss}
            # return scores, preds
            return preds


if __name__ == "__main__":
    from torch import nn

    from linkefl.common.factory import messenger_factory
    from linkefl.modelzoo.mlp import MLP, CutLayer
    from linkefl.util import num_input_nodes

    # 0. Set parameters
    _dataset_name = "tab_mnist"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _active_ips = ["localhost", "localhost"]
    _active_ports = [20000, 20001]
    _passive_ips = ["localhost", "localhost"]
    _passive_ports = [30000, 30001]
    _epochs = 10
    _batch_size = 100
    _learning_rate = 0.001
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _loss_fn = nn.CrossEntropyLoss()
    _num_workers = 1
    _random_state = None
    _saving_model = True
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _messengers = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=ac_ip,
            active_port=ac_port,
            passive_ip=pass_ip,
            passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in zip(
            _active_ips, _active_ports, _passive_ips, _passive_ports
        )
    ]

    # 1. Load datasets
    print("Loading dataset...")
    active_trainset = TorchDataset.buildin_dataset(
        dataset_name=_dataset_name,
        role=Const.ACTIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
    )
    active_testset = TorchDataset.buildin_dataset(
        dataset_name=_dataset_name,
        role=Const.ACTIVE_NAME,
        root="../data",
        train=False,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
    )
    # active_trainset = MediaDataset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_name=_dataset_name,
    #     root="../data",
    #     train=False,
    #     download=True,
    # )
    # active_testset = MediaDataset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_name=_dataset_name,
    #     root="../data",
    #     train=False,
    #     download=True,
    # )
    print("Done.")

    # 2. Create PyTorch models and optimizers
    weight = loss_reweight(active_trainset.labels)
    _loss_fn = nn.CrossEntropyLoss(weight=weight.to(_device))
    # _loss_fn = nn.CrossEntropyLoss()

    input_nodes = num_input_nodes(
        dataset_name=_dataset_name,
        role=Const.ACTIVE_NAME,
        passive_feat_frac=_passive_feat_frac,
    )
    # # mnist & fashion_mnist
    bottom_nodes = [input_nodes, 256, 128]
    cut_nodes = [128, 64]
    top_nodes = [64, 10]

    # criteo
    # bottom_nodes = [input_nodes, 15, 10]
    # cut_nodes = [10, 10]
    # top_nodes = [10, 2]

    # census
    # bottom_nodes = [input_nodes, 20, 10]
    # cut_nodes = [10, 10]
    # top_nodes = [10, 2]

    # epsilon
    # bottom_nodes = [input_nodes, 25, 10]
    # cut_nodes = [10, 10]
    # top_nodes = [10, 2]

    # credit
    # bottom_nodes = [input_nodes, 3, 3]
    # cut_nodes = [3, 3]
    # top_nodes = [3, 2]

    # default_credit
    # bottom_nodes = [input_nodes, 8, 5]
    # cut_nodes = [5, 5]
    # top_nodes = [5, 2]
    _bottom_model = MLP(
        bottom_nodes,
        activate_input=False,
        activate_output=True,
        random_state=_random_state,
    ).to(_device)
    # _bottom_model = ResNet18(in_channel=1).to(_device)
    _cut_layer = CutLayer(*cut_nodes, random_state=_random_state).to(_device)
    _top_model = MLP(
        top_nodes,
        activate_input=True,
        activate_output=False,
        random_state=_random_state,
    ).to(_device)
    _models = {"bottom": _bottom_model, "cut": _cut_layer, "top": _top_model}
    _optimizers = {
        name: torch.optim.SGD(
            model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
        )
        for name, model in _models.items()
    }

    # Initialize vertical NN protocol and start training
    print("Active party started, listening...")
    active_party = ActiveNeuralNetwork(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=_models,
        optimizers=_optimizers,
        loss_fn=_loss_fn,
        messengers=_messengers,
        logger=_logger,
        num_workers=_num_workers,
        device=_device,
        random_state=_random_state,
        saving_model=_saving_model,
    )
    active_party.train(active_trainset, active_testset)

    # Close messenger, finish training
    for msger_ in _messengers:
        msger_.close()
