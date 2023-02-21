import datetime
import time

import torch
from termcolor import colored
from torch.utils.data import DataLoader

from linkefl.common.const import Const
from linkefl.common.factory import (
    crypto_factory,
    messenger_factory,
    messenger_factory_multi_disconnection,
)
from linkefl.dataio import TorchDataset
from linkefl.util import num_input_nodes
from linkefl.vfl.nn.enc_layer import PassiveEncLayer
from linkefl.vfl.nn.model import CutLayer, MLPModel
from linkefl.vfl.nn.passive_new_disconnection import PassiveNeuralNetwork_disconnection


class PassiveNeuralNetwork_reconnection(PassiveNeuralNetwork_disconnection):
    def __init__(
        self,
        epochs,
        batch_size,
        learning_rate,
        models,
        optimizers,
        messenger,
        cryptosystem,
        *,
        precision=0.001,
        random_state=None,
        saving_model=False,
        model_path="./models",
    ):
        super(PassiveNeuralNetwork_reconnection, self).__init__(
            epochs,
            batch_size,
            learning_rate,
            models,
            optimizers,
            messenger,
            cryptosystem,
            precision=0.001,
            random_state=None,
            saving_model=False,
            model_path="./models",
        )

    def train(self, trainset, testset):
        assert isinstance(
            trainset, TorchDataset
        ), "trainset should be an instance of TorchDataset"
        assert isinstance(
            testset, TorchDataset
        ), "testset should be an instance of TorchDataset"
        train_dataloader = self._init_dataloader(trainset)
        test_dataloader = self._init_dataloader(testset)
        new_epoch = self.messenger.recv()
        print("Init Done!")

        for model in self.models.values():
            model.train()

        start_time = time.time()
        for epoch in range(new_epoch, self.epochs):
            print("Epoch: {}".format(epoch))
            for batch_idx, X in enumerate(train_dataloader):
                # print(f"batch: {batch_idx}")
                # 1. forward
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
                    plain_grad = torch.tensor(plain_grad).reshape(grad.shape)
                    bottom_outputs.backward(plain_grad)
                    self.optimizers["bottom"].step()
                else:
                    grad = self.messenger.recv()
                    passive_repr.backward(grad)
                    self.optimizers["cut"].step()
                    self.optimizers["bottom"].step()

                # break

            scores = self.validate(  # noqa: F841
                testset, existing_loader=test_dataloader
            )
            is_best = self.messenger.recv()
            if is_best:
                print(colored("Best model updated.", "red"))

        print(
            colored(
                "Total training and validation time: {}".format(
                    time.time() - start_time
                ),
                "red",
            )
        )


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "census"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = "localhost"
    active_port = 20000
    passive_ip = "localhost"
    passive_port = 30001
    _epochs = 200
    _batch_size = 100
    _learning_rate = 0.01
    _passive_in_nodes = 10
    _crypto_type = Const.PLAIN
    _key_size = 1024
    torch.manual_seed(1314)

    # 1. Load datasets
    print("Loading dataset...")
    passive_trainset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.PASSIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_testset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.PASSIVE_NAME,
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
        role=Const.PASSIVE_NAME,
        passive_feat_frac=passive_feat_frac,
    )
    # mnist & fashion_mnist
    # bottom_nodes = [input_nodes, 256, 128]
    # cut_nodes = [_passive_in_nodes, 64]

    # criteo
    # bottom_nodes = [input_nodes, 15, 10]
    # cut_nodes = [10, 10]

    # census
    bottom_nodes = [input_nodes, 20, 10]
    cut_nodes = [_passive_in_nodes, 8]
    bottom_model = MLPModel(bottom_nodes, activate_input=False, activate_output=True)
    cut_layer = CutLayer(*cut_nodes)
    _models = {"bottom": bottom_model, "cut": cut_layer}
    _optimizers = {
        name: torch.optim.SGD(model.parameters(), lr=_learning_rate)
        for name, model in _models.items()
    }

    # 3. Initialize messenger and cryptosystem
    _messenger = messenger_factory_multi_disconnection(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        model_type="NN",
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )
    _crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=100,
        gen_from_set=False,
    )

    # 4. Initialize vertical NN protocol and start fed training
    passive_party = PassiveNeuralNetwork_reconnection(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=_models,
        optimizers=_optimizers,
        messenger=_messenger,
        cryptosystem=_crypto,
    )
    passive_party.train(passive_trainset, passive_testset)

    # 5. Close messenger, finish training
    _messenger.close()
