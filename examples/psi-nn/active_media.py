import torch
from termcolor import colored
from torch import nn

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset
from linkefl.messenger import FastSocket
from linkefl.modelzoo import MLP, CutLayer, ResNet18
from linkefl.vfl.nn import ActiveNeuralNetwork

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_dir = "data"
    _dataset_name = "mnist"
    _active_ip = "localhost"
    _active_port = 20000
    _passive_ip = "localhost"
    _passive_port = 30000
    _epochs = 50
    _batch_size = 256
    _learning_rate = 0.001
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _saving_model = True
    _cut_nodes = [10, 10]
    _n_classes = 10
    _top_nodes = [10, _n_classes]
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _messenger = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip=_active_ip,
        active_port=_active_port,
        passive_ip=_passive_ip,
        passive_port=_passive_port,
    )

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
    bottom_model = ResNet18(in_channel=1).to(_device)
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
    # Initialize vertical NN protocol and start training
    active_party = ActiveNeuralNetwork(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=_models,
        optimizers=_optimizers,
        loss_fn=_loss_fn,
        messenger=_messenger,
        logger=_logger,
        device=_device,
        num_workers=1,
        val_freq=1,
        random_state=_random_state,
        saving_model=_saving_model,
    )
    active_party.train(active_trainset, active_testset)
    print(colored("3. Active party finished vfl_nn training.", "red"))

    # 3. Close messenger, finish training
    _messenger.close()
