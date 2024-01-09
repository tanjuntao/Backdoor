import torch
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset
from linkefl.modelzoo import *
from linkefl.vfl.nn import ActiveNeuralNetwork

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_dir = "../data"
    _dataset_name = "cifar10"
    _epochs = 50
    _batch_size = 128
    _learning_rate = 0.1
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = None
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _cut_nodes = [10, 10]
    _n_classes = 10
    _top_nodes = [10, _n_classes]
    _logger = logger_factory(role=Const.ACTIVE_NAME)

    # 1. Load dataset
    active_trainset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root=_dataset_dir,
        train=True,
        download=True,
    )
    active_testset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root=_dataset_dir,
        train=False,
        download=True,
    )
    print(colored("1. Finish loading dataset.", "red"))

    # 2. VFL training
    print(colored("2. Active party started training...", "red"))
    bottom_model = VGG11(in_channel=3).to(_device)
    cut_layer = CutLayer(*_cut_nodes, random_state=_random_state).to(_device)
    top_model = MLP(
        _top_nodes,
        activate_input=True,
        activate_output=False,
        random_state=_random_state,
    ).to(_device)
    _models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}
    _optimizers = {
        name: torch.optim.SGD(
            model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
        )
        for name, model in _models.items()
    }
    schedulers = {
        name: CosineAnnealingLR(optimizer=optimizer, T_max=_epochs, eta_min=0)
        for name, optimizer in _optimizers.items()
    }
    # Initialize vertical NN protocol and start training
    active_party = ActiveNeuralNetwork(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=_models,
        optimizers=_optimizers,
        loss_fn=_loss_fn,
        messengers=None,
        logger=_logger,
        device=_device,
        num_workers=1,
        val_freq=1,
        random_state=_random_state,
        saving_model=False,
        schedulers=schedulers,
    )
    active_party.train_alone(active_trainset, active_testset)
    active_party.validate_alone(active_testset)
    print(colored("3. Active party finished vfl_nn training.", "red"))
