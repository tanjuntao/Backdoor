import datetime
import os
import pathlib
import pickle
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
from linkefl.common.factory import partial_crypto_factory
from linkefl.common.log import GlobalLogger
from linkefl.dataio import MediaDataset, TorchDataset  # noqa: F403
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *  # noqa
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
        args=None,
        start_epoch=0,
        best_acc=0,
    ):
        self.epochs = epochs
        self.start_epoch = start_epoch
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
        self.args = args
        self.best_acc = best_acc

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
        self,
        dataset,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        bs=None,
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
        best_acc, best_auc = self.best_acc, 0.0  # TODO: best_auc as init param
        if self.topk > 1:
            best_topk_acc = 0.0
        train_acc_records, valid_acc_records = [], []
        train_auc_records, valid_auc_records = [], []
        train_loss_records, valid_loss_records = [], []
        history_embedding_norms = []
        history_embedding_grad_norms = []
        history_bottom_ouput_norms = []
        n_classes = torch.numel(torch.unique(trainset.labels))
        ##### Main Loop ######
        for epoch in range(self.epochs):
            # layer_importance = {}
            embedding_norm = {label: [] for label in range(n_classes)}
            embedding_grad_norm = {label: [] for label in range(n_classes)}
            bottom_output_norm = {label: [] for label in range(n_classes)}
            train_loss, correct, total = 0, 0, 0
            if self.topk > 1:
                topk_correct = 0
            for model in self.models.values():
                model.train()
            self.logger.log(f"Epoch {epoch} started...")
            print(f"Epoch: {epoch}, Actual Epoch: {epoch + self.start_epoch}")
            torch.manual_seed(epoch)  # fix dataloader batching order when shuffle=True
            for batch_idx, (X, y) in enumerate(train_dataloader):
                # 1. forward
                X = X.to(self.device)
                y = y.to(self.device)

                if self.args.agg == "add":
                    active_embedding = self.models["bottom"](X)
                    active_repr = self.models["cut"](active_embedding)
                elif self.args.agg == "concat":
                    active_embedding = self.models["bottom"](X)
                    active_repr = active_embedding
                else:
                    raise ValueError(f"{self.args.agg} is not supported.")

                # Embedding norms
                batch_embedding_norm = torch.norm(active_repr.data, p=2, dim=1)
                for label in range(n_classes):
                    embedding_norm[label].extend(
                        batch_embedding_norm[y == label].cpu().tolist()
                    )

                batch_output_norm = torch.norm(active_embedding.data, p=2, dim=1)
                for label in range(n_classes):
                    bottom_output_norm[label].extend(
                        batch_output_norm[y == label].cpu().tolist()
                    )

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
                    if self.args.agg == "add":
                        top_input += passive_repr
                    elif self.args.agg == "concat":
                        top_input = torch.concat((top_input, passive_repr), dim=1)

                # top_input = active_repr.data + passive_repr
                top_input = top_input.requires_grad_()
                logits = self.models["top"](top_input)
                loss = self.loss_fn(logits, y)

                # 2. backward
                for optmizer in self.optimizers.values():
                    optmizer.zero_grad()
                loss.backward()

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

                # Embedding gradient norms
                batch_embedding_grad_norm = torch.norm(top_input.grad.data, p=2, dim=1)
                for label in range(n_classes):
                    embedding_grad_norm[label].extend(
                        batch_embedding_grad_norm[y == label].cpu().tolist()
                    )

                # update active party's top model, cut layer and bottom model
                if self.args.agg == "add":
                    active_repr_grad = top_input.grad
                elif self.args.agg == "concat":
                    active_repr_grad = top_input.grad[:, : active_repr.shape[1]]
                active_repr.backward(active_repr_grad)
                for optmizer in self.optimizers.values():
                    optmizer.step()

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
                            epoch=epoch + self.start_epoch,
                        )
                if self.save_every_epoch:
                    TorchModelIO.save(
                        self.models,
                        self.model_dir,
                        f"active_epoch_{epoch + self.start_epoch}.model",
                        epoch=epoch + self.start_epoch,
                    )
                    if not os.path.exists(f"{self.model_dir}/optim"):
                        os.mkdir(f"{self.model_dir}/optim")
                    torch.save(
                        self.optimizers["bottom"].state_dict(),
                        f"{self.model_dir}/optim/active_optim_bottom_epoch_{epoch + self.start_epoch}.pth",
                    )
                    torch.save(
                        self.optimizers["cut"].state_dict(),
                        f"{self.model_dir}/optim/active_optim_cut_epoch_{epoch + self.start_epoch}.pth",
                    )
                    torch.save(
                        self.optimizers["top"].state_dict(),
                        f"{self.model_dir}/optim/active_optim_top_epoch_{epoch + self.start_epoch}.pth",
                    )

                for msger in self.messengers:
                    msger.send(is_best)

            embedding_norm = {
                label: sum(values) / len(values)
                for label, values in embedding_norm.items()
            }
            embedding_grad_norm = {
                label: sum(values) / len(values)
                for label, values in embedding_grad_norm.items()
            }
            bottom_output_norm = {
                label: sum(values) / len(values)
                for label, values in bottom_output_norm.items()
            }
            history_embedding_norms.append(embedding_norm)
            history_embedding_grad_norms.append(embedding_grad_norm)
            history_bottom_ouput_norms.append(bottom_output_norm)

        # save history embedding norms for vis
        with open(f"{self.model_dir}/history_embedding_norms.pkl", "wb") as f:
            pickle.dump(history_embedding_norms, f)
        with open(f"{self.model_dir}/history_embedding_grad_norms.pkl", "wb") as f:
            pickle.dump(history_embedding_grad_norms, f)
        with open(f"{self.model_dir}/history_bottom_ouput_norms.pkl", "wb") as f:
            pickle.dump(history_bottom_ouput_norms, f)
        with open(f"{self.model_dir}/history_test_acc.pkl", "wb") as f:
            pickle.dump(valid_acc_records, f)

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
            print(
                colored(
                    "Best testing topK accuracy: {:.5f}".format(best_topk_acc), "red"
                )
            )
        print(colored("Best testing auc: {:.5f}".format(best_auc), "red"))
        self.logger.log(f"elapsed time: {time.time() - start_time}")
        self.logger.log("Best testing accuracy: {:.5f}".format(best_acc))
        self.logger.log("Best testing auc: {:.5f}".format(best_auc))

        return {
            "best_acc": best_acc,
            "train_loss_records": train_loss_records,
            "valid_loss_records": valid_loss_records,
            "train_acc_records": train_acc_records,
            "valid_acc_records": valid_acc_records,
        }

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
                if self.args.agg == "add":
                    active_repr = self.models["cut"](self.models["bottom"](X))
                elif self.args.agg == "concat":
                    active_repr = self.models["bottom"](X)
                top_input = active_repr
                for idx, msger in enumerate(self.messengers):
                    passive_repr = msger.recv().to(self.device)
                    if self.args.agg == "add":
                        top_input += passive_repr
                    elif self.args.agg == "concat":
                        top_input = torch.concat((active_repr, passive_repr), dim=1)
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

    def validate_attack(
        self,
        testset: TorchDataset,
        *,
        existing_loader: Optional[DataLoader] = None,
    ):
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
        predictions = np.array([])
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                if self.args.agg == "add":
                    active_repr = self.models["cut"](self.models["bottom"](X))
                elif self.args.agg == "concat":
                    active_repr = self.models["bottom"](X)
                top_input = active_repr
                for idx, msger in enumerate(self.messengers):
                    passive_repr = msger.recv().to(self.device)
                    if self.args.agg == "add":
                        top_input += passive_repr
                    elif self.args.agg == "concat":
                        top_input = torch.concat((top_input, passive_repr), dim=1)
                # top_input = active_repr + passive_repr
                logits = self.models["top"](top_input)
                labels = np.append(labels, y.cpu().numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(logits[:, 1]).cpu().numpy())
                test_loss += self.loss_fn(logits, y).item()
                _, predicted = logits.max(1)
                predictions = np.append(
                    predictions, predicted.cpu().numpy().astype(np.int32)
                )
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

            # calculate per-class backdoor attack accuracy
            print("=======> Main Task Accuracy")
            for category in range(n_classes):
                category_num_samples = len(labels[labels == category])
                category_correct = (
                    (labels[labels == category] == predictions[labels == category])
                    .astype(np.int32)
                    .sum()
                )
                print(f"Class {category}: {category_correct / category_num_samples}")

            print("=======> Backdoor Attack Accuracy")
            asr = (predictions == self.args.target).astype(np.int32).mean()
            print(f"Total backdoor attack success rate: {asr}")
            for category in range(n_classes):
                category_num_samples = len(labels[labels == category])
                backdoor_success = (
                    (predictions[labels == category] == self.args.target)
                    .astype(np.int32)
                    .sum()
                )
                print(f"Class {category}: {backdoor_success / category_num_samples}")

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
        train_dataloader = self._init_dataloader(
            trainset, shuffle=True, num_workers=2, bs=self.batch_size
        )
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
                if self.args.agg == "add":
                    logits = self.models["top"](
                        self.models["cut"](self.models["bottom"](X))
                    )
                elif self.args.agg == "concat":
                    logits = self.models["top"](self.models["bottom"](X))

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
            print(
                colored(
                    "Best testing topK accuracy: {:.5f}".format(best_topk_acc), "red"
                )
            )
        print(colored("Best testing auc: {:.5f}".format(best_auc), "red"))
        if self.topk > 1:
            return (best_acc, best_topk_acc)
        else:
            return (best_acc,)

    def validate_alone(
        self,
        testset: TorchDataset,
        *,
        existing_loader: Optional[DataLoader] = None,
        cal_topk_acc=False,
        topk_confident=50,
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
        if cal_topk_acc:
            softmax_probs = np.array([])
            predictions = np.array([])
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                embedding = self.models["bottom"](X)
                if total_embeddings is None:
                    total_embeddings = torch.zeros(len(testset), embedding.size(1)).to(
                        self.device
                    )
                index = torch.arange(start_idx, start_idx + X.size(0)).to(self.device)
                total_embeddings.index_copy_(0, index, embedding)
                start_idx = start_idx + X.size(0)

                if self.args.agg == "add":
                    logits = self.models["top"](self.models["cut"](embedding))
                elif self.args.agg == "concat":
                    logits = self.models["top"](embedding)
                if cal_topk_acc:
                    softmax_probs = np.append(
                        softmax_probs,
                        nn.functional.softmax(logits, dim=1).cpu().numpy(),
                    )
                labels = np.append(labels, y.cpu().numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(logits[:, 1]).cpu().numpy())
                test_loss += self.loss_fn(logits, y).item()
                _, predicted = logits.max(1)
                if cal_topk_acc:
                    predictions = np.append(
                        predictions, predicted.cpu().numpy().astype(np.int32)
                    )
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

            # calculate accuracy of TopK most confident samples
            if cal_topk_acc:
                softmax_probs = softmax_probs.reshape(len(labels), -1)
                target_indices = np.where(predictions == self.args.target)[
                    0
                ]  # np.where returns tuple, first element is index
                target_probs = softmax_probs[target_indices][:, self.args.target]
                target_ground_truths = labels[target_indices]
                topk_indices = np.argsort(target_probs)[
                    -topk_confident:
                ]  # ascending order
                topk_acc = (
                    (target_ground_truths[topk_indices] == self.args.target)
                    .astype(np.int32)
                    .mean()
                )
                scores["topk_confident_acc"] = round(float(topk_acc), 5)
                topk_original_indices = target_indices[topk_indices]
                scores["topk_original_indices"] = topk_original_indices  # np array

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
