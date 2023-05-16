import datetime
import os
import pathlib
import time
from typing import Dict, Optional

import torch
from termcolor import colored
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from linkefl.base import BaseCryptoSystem, BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
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
        rank: int = 1,
        num_workers: int = 1,
        val_freq: int = 1,
        device: str = "cpu",
        encode_precision: float = 0.001,
        random_state: Optional[int] = None,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = models
        self.optimizers = optimizers
        self.messenger = messenger
        self.cryptosystem = cryptosystem
        self.logger = logger
        self.rank = rank
        self.num_workers = num_workers
        self.val_freq = val_freq
        self.device = device
        self.encode_precision = encode_precision
        self.random_state = random_state
        if random_state is not None:
            torch.random.manual_seed(random_state)
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

    def _init_dataloader(self, dataset, shuffle=False):
        bs = dataset.n_samples if self.batch_size == -1 else self.batch_size
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle)
        return dataloader

    def train(self, trainset: TorchDataset, testset: TorchDataset) -> None:
        assert isinstance(
            trainset, TorchDataset
        ), "trainset should be an instance of TorchDataset"
        assert isinstance(
            testset, TorchDataset
        ), "testset should be an instance of TorchDataset"
        train_dataloader = self._init_dataloader(trainset)
        test_dataloader = self._init_dataloader(testset)

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
            print(f"Epoch: {epoch}")
            self.logger.log(f"Epoch {epoch} started...")
            for batch_idx, X in enumerate(train_dataloader):
                # 1. forward
                X = X.to(self.device)
                bottom_outputs = self.models["bottom"](X)
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    passive_repr = self.enc_layer.fed_forward(bottom_outputs)
                else:
                    passive_repr = self.models["cut"](bottom_outputs)
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
                    passive_repr.backward(grad)
                    self.optimizers["cut"].step()
                    self.optimizers["bottom"].step()

            # validate model
            if (epoch + 1) % self.val_freq == 0:
                self.validate(testset, existing_loader=test_dataloader)
                self.validate(trainset, existing_loader=train_dataloader)
                is_best = self.messenger.recv()
                if is_best:
                    print(colored("Best model updated.", "red"))
                    if self.saving_model:
                        TorchModelIO.save(
                            self.models,
                            self.model_dir,
                            self.model_name,
                            epoch=epoch,
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
            bs = testset.n_samples if self.batch_size == -1 else self.batch_size
            dataloader = DataLoader(testset, batch_size=bs, shuffle=False)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        with torch.no_grad():
            for batch, X in enumerate(dataloader):
                X = X.to(self.device)
                bottom_outputs = self.models["bottom"](X)
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    passive_repr = self.enc_layer.fed_forward(bottom_outputs)
                else:
                    passive_repr = self.models["cut"](bottom_outputs)
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


if __name__ == "__main__":
    from linkefl.common.factory import crypto_factory, messenger_factory
    from linkefl.modelzoo.mlp import MLP, CutLayer
    from linkefl.util import num_input_nodes

    # 0. Set parameters
    _dataset_name = "tab_mnist"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _active_ip = "localhost"
    _active_port = 20000
    _passive_ip = "localhost"
    _passive_port = 30000
    _epochs = 10
    _batch_size = 100
    _learning_rate = 0.001
    _crypto_type = Const.PLAIN
    _key_size = 1024
    _num_workers = 1
    _random_state = None
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _saving_model = True
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    _messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=_active_ip,
        active_port=_active_port,
        passive_ip=_passive_ip,
        passive_port=_passive_port,
    )
    _crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=100,
        gen_from_set=False,
    )

    # 1. Load datasets
    print("Loading dataset...")
    passive_trainset = TorchDataset.buildin_dataset(
        dataset_name=_dataset_name,
        role=Const.PASSIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
    )
    passive_testset = TorchDataset.buildin_dataset(
        dataset_name=_dataset_name,
        role=Const.PASSIVE_NAME,
        root="../data",
        train=False,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
    )
    # passive_trainset = MediaDataset(
    #     role=Const.PASSIVE_NAME,
    #     dataset_name=_dataset_name,
    #     root="../data",
    #     train=False,
    #     download=True,
    # )
    # passive_testset = MediaDataset(
    #     role=Const.PASSIVE_NAME,
    #     dataset_name=_dataset_name,
    #     root="../data",
    #     train=False,
    #     download=True,
    # )
    # passive_trainset = TorchDataset.feature_split(passive_trainset, n_splits=2)[0]
    # passive_testset = TorchDataset.feature_split(passive_testset, n_splits=2)[0]
    print("Done.")

    # 2. Create PyTorch models and optimizers
    input_nodes = num_input_nodes(
        dataset_name=_dataset_name,
        role=Const.PASSIVE_NAME,
        passive_feat_frac=_passive_feat_frac,
    )
    # # mnist & fashion_mnist
    bottom_nodes = [input_nodes, 256, 128]
    cut_nodes = [128, 64]

    # criteo
    # bottom_nodes = [input_nodes, 15, 10]
    # cut_nodes = [10, 10]

    # census
    # bottom_nodes = [input_nodes, 20, 10]
    # cut_nodes = [10, 10]

    # epsilon
    # bottom_nodes = [input_nodes, 25, 10]
    # cut_nodes = [10, 10]

    # credit
    # bottom_nodes = [input_nodes, 3, 3]
    # cut_nodes = [3, 3]

    # default_credit
    # bottom_nodes = [input_nodes, 8, 5]
    # cut_nodes = [5, 5]
    _bottom_model = MLP(
        bottom_nodes,
        activate_input=False,
        activate_output=True,
        random_state=_random_state,
    ).to(_device)
    # bottom_model = ResNet18(in_channel=1).to(_device)
    _cut_layer = CutLayer(*cut_nodes, random_state=_random_state).to(_device)
    _models = {"bottom": _bottom_model, "cut": _cut_layer}
    _optimizers = {
        name: torch.optim.SGD(
            model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
        )
        for name, model in _models.items()
    }

    # 3. Initialize vertical NN protocol and start fed training
    passive_party = PassiveNeuralNetwork(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=_models,
        optimizers=_optimizers,
        messenger=_messenger,
        cryptosystem=_crypto,
        logger=_logger,
        num_workers=_num_workers,
        device=_device,
        random_state=_random_state,
        saving_model=_saving_model,
    )
    passive_party.train(passive_trainset, passive_testset)

    # 4. Close messenger, finish training
    _messenger.close()
