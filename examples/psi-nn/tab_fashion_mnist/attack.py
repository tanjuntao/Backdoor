import torch
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import TorchDataset
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo.mlp import MLP, CutLayer
from linkefl.util import num_input_nodes
from linkefl.vfl.nn import ActiveNeuralNetwork
from linkefl.vfl.nn.active import loss_reweight

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_name = "tab_fashion_mnist"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _epochs = 50
    _batch_size = 8
    _learning_rate = 0.01
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _loss_fn = nn.CrossEntropyLoss()
    _num_workers = 1
    _random_state = None
    _device = "cpu"

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
    finetune_dataset = TorchDataset.buildin_dataset(
        dataset_name=_dataset_name,
        role=Const.ACTIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
        fine_tune=True,
        fine_tune_per_class=32,
    )
    print("Done.")
    print(finetune_dataset.n_samples, finetune_dataset.n_features)

    # 2. Create PyTorch models and optimizers
    weight = loss_reweight(active_trainset.labels)
    _loss_fn = nn.CrossEntropyLoss(weight=weight.to(_device))

    input_nodes = num_input_nodes(
        dataset_name=_dataset_name,
        role=Const.ACTIVE_NAME,
        passive_feat_frac=_passive_feat_frac,
    )
    # # mnist & fashion_mnist
    bottom_nodes = [input_nodes, 256, 128, 128]
    cut_nodes = [128, 64]
    top_nodes = [64, 10]

    # _bottom_model = MLP(
    #     bottom_nodes,
    #     activate_input=False,
    #     activate_output=True,
    #     random_state=_random_state,
    # ).to(_device)
    _bottom_model = TorchModelIO.load(
        "../models/tab_fashion_mnist", "VFL_active.model"
    )["model"]["bottom"].to(_device)
    _bottom_model.sequential[0] = nn.Linear(
        in_features=392, out_features=256, bias=True
    ).to(_device)
    _bottom_model.sequential[3] = nn.Linear(
        in_features=256, out_features=128, bias=True
    ).to(_device)
    _bottom_model.sequential[6] = nn.Linear(
        in_features=128, out_features=128, bias=True
    ).to(_device)
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
    _optimizers["bottom"] = torch.optim.SGD(
        _bottom_model.parameters(),
        lr=_learning_rate / 100,
        momentum=0.9,
        weight_decay=5e-4,
    )
    schedulers = {
        name: CosineAnnealingLR(optimizer=optimizer, T_max=_epochs, eta_min=0)
        for name, optimizer in _optimizers.items()
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
        messengers=None,
        logger=_logger,
        num_workers=_num_workers,
        device=_device,
        random_state=_random_state,
        saving_model=False,
        schedulers=schedulers,
        model_dir="../models/tab_fashion_mnist",
        model_name="attack.model",
    )
    active_party.train_alone(finetune_dataset, active_testset)
    active_party.validate_alone(active_testset)
    print(colored("3. Active party finished vfl_nn training.", "red"))
