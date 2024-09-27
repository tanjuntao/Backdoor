import datetime
import os
import pathlib
import time
from typing import Any, Dict, Optional

import torch
from termcolor import colored
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from linkefl.base import BaseCryptoSystem, BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import MediaDataset, TorchDataset  # noqa: F403
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *  # noqa: F403
from linkefl.vfl.nn.enc_layer import PassiveEncLayer


class PassiveNeuralNetwork(BaseModelComponent):
    def __init__(
        self,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Optimizer],
        messenger: BaseMessenger,
        cryptosystem: BaseCryptoSystem,
        logger: GlobalLogger,
        schedulers: Optional[Dict[str, Any]] = None,
        rank: int = 1,
        num_workers: int = 1,
        val_freq: int = 1,
        device: str = "cpu",
        encode_precision: float = 0.001,
        random_state: Optional[int] = None,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        args=None,
        start_epoch=0,
    ):
        self.args = args
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = models
        self.optimizers = optimizers
        self.messenger = messenger
        self.cryptosystem = cryptosystem
        self.logger = logger
        self.schedulers = schedulers
        self.rank = rank
        self.num_workers = num_workers
        self.val_freq = val_freq
        self.device = device
        self.encode_precision = encode_precision
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.saving_model = saving_model
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
                        role=Const.PASSIVE_NAME,
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
        self.enc_layer = None

    def _sync_pubkey(self):
        print("Training protocol started.")
        print("Sending public key...")
        public_key, crypto_type = self.cryptosystem.pub_key, self.cryptosystem.type
        self.messenger.send([public_key, crypto_type])
        print("Done.")

    def _init_dataloader(
        self, dataset, shuffle=False, num_workers=1, persistent_workers=True
    ):
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

        self._sync_pubkey()
        if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
            self.messenger.send(self.models["cut"].in_nodes)
            self.enc_layer = PassiveEncLayer(
                in_nodes=self.models["cut"].in_nodes,
                out_nodes=self.models["cut"].out_nodes,
                eta=self.learning_rate,
                messenger=self.messenger,
                cryptosystem=self.cryptosystem,
                num_workers=self.num_workers,
                random_state=self.random_state,
                encode_precision=self.encode_precision,
            )

        for model in self.models.values():
            model.train()

        start_time = time.time()
        for epoch in range(self.epochs):
            torch.manual_seed(epoch)  # fix dataloader batching order when shuffle=True
            for model in self.models.values():
                model.train()
            print(f"Epoch: {epoch}, Actual Epoch: {epoch + self.start_epoch}")
            self.logger.log(f"Epoch {epoch} started...")
            for batch_idx, (X, _) in enumerate(train_dataloader):
                # 1. forward
                X = X.to(self.device)
                bottom_outputs = self.models["bottom"](X)
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    passive_repr = self.enc_layer.fed_forward(bottom_outputs)
                else:
                    if self.args.agg == "add":
                        passive_repr = self.models["cut"](bottom_outputs)
                    elif self.args.agg == "concat":
                        passive_repr = bottom_outputs
                self.messenger.send(passive_repr.data)

                # 2. backward
                for optmizer in self.optimizers.values():
                    optmizer.zero_grad()
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    self.enc_layer.fed_backward()
                    grad = self.messenger.recv()
                    plain_grad = self.cryptosystem.decrypt_vector(grad.flatten())
                    plain_grad = (
                        torch.tensor(plain_grad).reshape(grad.shape).to(self.device)
                    )
                    bottom_outputs.backward(plain_grad)
                    self.optimizers["bottom"].step()
                else:
                    grad = self.messenger.recv().to(self.device)
                    if self.args.agg == "add":
                        passive_repr.backward(grad)
                    elif self.args.agg == "concat":
                        passive_repr.backward(grad[:, -passive_repr.shape[1] :])
                    self.optimizers["cut"].step()
                    self.optimizers["bottom"].step()

            # update learning rate scheduler
            if self.schedulers is not None:
                for scheduler in self.schedulers.values():
                    scheduler.step()

            # validate model
            if (epoch + 1) % self.val_freq == 0:
                self.validate(testset, existing_loader=test_dataloader)
                is_best = self.messenger.recv()
                if is_best:
                    print(colored("Best model updated.", "red"))
                    if self.saving_model:
                        TorchModelIO.save(
                            self.models,
                            self.model_dir,
                            self.model_name,
                            epoch=epoch + self.start_epoch,
                        )
                if self.saving_model:
                    TorchModelIO.save(
                        self.models,
                        self.model_dir,
                        f"passive_epoch_{epoch + self.start_epoch}.model",
                        epoch=epoch + self.start_epoch,
                    )
                    if not os.path.exists(f"{self.model_dir}/optim"):
                        os.mkdir(f"{self.model_dir}/optim")
                    torch.save(
                        self.optimizers["bottom"].state_dict(),
                        f"{self.model_dir}/optim/passive_optim_bottom_epoch_{epoch + self.start_epoch}.pth",
                    )
                    torch.save(
                        self.optimizers["cut"].state_dict(),
                        f"{self.model_dir}/optim/passive_optim_cut_epoch_{epoch + self.start_epoch}.pth",
                    )

        # close pool
        if self.enc_layer is not None:
            self.enc_layer.close_pool()

        print(colored(f"elapsed time: {time.time() - start_time}", "red"))
        self.logger.log(
            "Total training and validation time: {}".format(time.time() - start_time)
        )

    def validate(
        self,
        testset: TorchDataset,
        existing_loader: Optional[DataLoader] = None,
    ) -> None:
        if existing_loader is None:
            dataloader = self._init_dataloader(testset, num_workers=2)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        with torch.no_grad():
            for batch, (X, _) in enumerate(dataloader):
                X = X.to(self.device)
                bottom_outputs = self.models["bottom"](X)
                if self.args.agg == "add":
                    passive_repr = self.models["cut"](bottom_outputs)
                elif self.args.agg == "concat":
                    passive_repr = bottom_outputs

                self.messenger.send(passive_repr)

    def validate_attack(
        self,
        testset: TorchDataset,
        existing_loader: Optional[DataLoader] = None,
        trigger_embedding=None,
    ) -> None:
        if existing_loader is None:
            dataloader = self._init_dataloader(testset, num_workers=2)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        with torch.no_grad():
            for batch, (X, _) in enumerate(dataloader):
                if trigger_embedding is None:
                    X = X.to(self.device)
                    bottom_outputs = self.models["bottom"](X)
                else:
                    bottom_outputs = trigger_embedding.repeat(X.size(0), 1)
                if self.args.agg == "add":
                    passive_repr = self.models["cut"](bottom_outputs)
                elif self.args.agg == "concat":
                    passive_repr = bottom_outputs
                self.messenger.send(passive_repr)

    def fit(
        self,
        trainset: TorchDataset,
        validset: TorchDataset,
        role: str = Const.PASSIVE_NAME,
    ) -> None:
        self.train(trainset, validset)

    def score(self, testset: TorchDataset, role: str = Const.PASSIVE_NAME) -> None:
        return self.validate(testset)

    @staticmethod
    def online_inference(
        dataset: TorchDataset,
        messenger: BaseMessenger,
        logger: GlobalLogger,
        model_dir: str,
        model_name: str,
        role: str = Const.PASSIVE_NAME,
    ):
        models: dict = TorchModelIO.load(model_dir, model_name)["model"]
        for model in models.values():
            model.eval()
        dataloader = DataLoader(dataset, batch_size=dataset.n_samples, shuffle=False)

        with torch.no_grad():
            for batch, X in enumerate(dataloader):
                bottom_outputs = models["bottom"](X)
                passive_repr = models["cut"](bottom_outputs)
                messenger.send(passive_repr)

    def validate_alone(
        self,
        testset: TorchDataset,
        existing_loader: Optional[DataLoader] = None,
    ) -> None:
        if existing_loader is None:
            dataloader = self._init_dataloader(testset, num_workers=2)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        total_embeddings = None
        start_idx = 0
        with torch.no_grad():
            for batch_idx, (X, _) in enumerate(dataloader):
                X = X.to(self.device)
                # no matter what agg() is, return the bottom model output as embedding
                embedding = self.models["bottom"](X)
                # embedding = self.models["cut"](embedding)
                if total_embeddings is None:
                    total_embeddings = torch.zeros(len(testset), embedding.size(1)).to(
                        self.device
                    )
                index = torch.arange(start_idx, start_idx + X.size(0)).to(self.device)
                total_embeddings.index_copy_(0, index, embedding)
                start_idx = start_idx + X.size(0)

        return total_embeddings
