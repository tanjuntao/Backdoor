import torch
from args_parser import get_args, get_mask_layers
from mask import layer_masking
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *
from linkefl.vfl.nn import ActiveNeuralNetwork

args = get_args()

if __name__ == "__main__":
    # Set params
    data_prefix = ".."
    _epochs = 50
    _learning_rate = 0.01
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = None
    _device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    if args.dataset in ("cifar10", "cinic10"):
        if args.dataset == "cifar10":
            _dataset_dir = f"{data_prefix}/data"
        else:
            _dataset_dir = f"{data_prefix}/data/CINIC10"
        topk = 1
        _batch_size = 4
        _cut_nodes = [10, 10]
        _n_classes = 10
        _top_nodes = [10, _n_classes]
    elif args.dataset == "cifar100":
        _dataset_dir = f"{data_prefix}/data"
        topk = 5
        _batch_size = 8
        _cut_nodes = [100, 100]
        _n_classes = 100
        _top_nodes = [100, _n_classes]
    elif args.dataset in ("mnist", "fashion_mnist", "svhn"):
        _batch_size = 4
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [10, 10]
        _n_classes = 10
        _top_nodes = [10, _n_classes]
    elif args.dataset in ("tab_mnist", "tab_fashion_mnist"):
        _batch_size = 4
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [128, 64]
        _n_classes = 10
        _top_nodes = [64, _n_classes]
        if args.world_size == 2:
            pass
        elif args.world_size == 4:
            pass
        elif args.world_size == 6:
            _cut_nodes = [96, 64]
        elif args.world_size == 8:
            _cut_nodes = [64, 32]
            _top_nodes = [32, 10]
        elif args.world_size == 10:
            _cut_nodes = [64, 32]
            _top_nodes = [32, 10]
        else:
            raise ValueError(f"{args.world_size} is not supported.")
    elif args.dataset == "criteo":
        _batch_size = 4
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [10, 10]
        _n_classes = 2
        _top_nodes = [10, _n_classes]
    else:
        raise ValueError(f"{args.dataset} is not valid dataset.")

    # Load dataset
    if args.model in ("resnet18", "vgg13", "lenet"):
        active_testset = MediaDataset(
            role=Const.ACTIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=False,
            download=True,
            world_size=args.world_size,
            rank=args.rank,
        )
        fine_tune_trainset = MediaDataset(
            role=Const.ACTIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=True,
            download=True,
            fine_tune=True,
            fine_tune_per_class=args.per_class,
            world_size=args.world_size,
            rank=args.rank,
        )
    elif args.model == "mlp":
        _passive_feat_frac = 1 - 1 / args.world_size
        _feat_perm_option = Const.SEQUENCE
        active_testset = TorchDataset.buildin_dataset(
            dataset_name=args.dataset,
            role=Const.ACTIVE_NAME,
            root=_dataset_dir,
            train=False,
            download=True,
            passive_feat_frac=_passive_feat_frac,
            feat_perm_option=_feat_perm_option,
            seed=_random_state,
        )
        fine_tune_trainset = TorchDataset.buildin_dataset(
            dataset_name=args.dataset,
            role=Const.ACTIVE_NAME,
            root=_dataset_dir,
            train=True,
            download=True,
            passive_feat_frac=_passive_feat_frac,
            feat_perm_option=_feat_perm_option,
            seed=_random_state,
            fine_tune=True,
            fine_tune_per_class=args.per_class,
        )
    print(colored("1. Finish loading dataset.", "red"))

    # Load model
    bottom_model = TorchModelIO.load(
        f"checkpoints/{args.model}_{args.dataset}_world{args.world_size}_attempt{args.attempt}",
        f"{args.rank}.model",
    )["model"]["bottom"].to(_device)
    if args.scratch:
        if args.model == "resnet18":
            bottom_model = ResNet18(in_channel=3, num_classes=_n_classes).to(_device)
        elif args.model == "vgg13":
            bottom_model = VGG13(in_channel=3, num_classes=_n_classes).to(_device)
        elif args.model == "lenet":
            in_channel = 1
            if args.dataset == "svhn":
                in_channel = 3
            bottom_model = LeNet(in_channel=in_channel, num_classes=_n_classes).to(
                _device
            )
        elif args.model == "mlp":
            if args.world_size == 2:
                bottom_nodes = [392, 256, 128, 128]
            elif args.world_size == 4:
                bottom_nodes = [196, 164, 128, 128]
            elif args.world_size == 6:
                bottom_nodes = [131, 128, 96, 96]
                _cut_nodes = [96, 64]
            elif args.world_size == 8:
                bottom_nodes = [98, 64, 64]
                _cut_nodes = [64, 32]
                _top_nodes = [32, 10]
            elif args.world_size == 10:
                bottom_nodes = [79, 64, 64]
                _cut_nodes = [64, 32]
                _top_nodes = [32, 10]
            else:
                raise ValueError(f"{args.world_size} is not supported.")
            # input_nodes = num_input_nodes(
            #     dataset_name=args.dataset,
            #     role=Const.ACTIVE_NAME,
            #     passive_feat_frac=_passive_feat_frac,
            # )
            # if args.dataset in ("tab_mnist", "tab_fashion_mnist"):
            #     bottom_nodes = [input_nodes, 256, 128, 128]
            # else:
            #     bottom_nodes = [input_nodes, 15, 10]
            bottom_model = MLP(
                bottom_nodes,
                activate_input=False,
                activate_output=True,
                random_state=_random_state,
            ).to(_device)
        else:
            raise ValueError(f"{args.model} is not an valid model type.")
    bottom_model = layer_masking(
        model_type=args.model,
        bottom_model=bottom_model,
        mask_layers=get_mask_layers(),
        device=_device,
        dataset=args.dataset,
        world_size=args.world_size,
        rank=args.rank,
    )
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
    _optimizers["bottom"] = torch.optim.SGD(
        bottom_model.parameters(),
        lr=_learning_rate / 100,
        momentum=0.9,
        weight_decay=5e-4,
    )
    schedulers = {
        name: CosineAnnealingLR(optimizer=optimizer, T_max=_epochs, eta_min=0)
        for name, optimizer in _optimizers.items()
    }

    # Model training
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
        topk=topk,
        random_state=_random_state,
        saving_model=False,
        schedulers=schedulers,
    )
    active_party.train_alone(fine_tune_trainset, active_testset)
    print(colored("3. Active party finished vfl_nn training.", "red"))


# command
# python3 attack.py --model mlp --dataset tab_mnist --gpu 0 --world_size 6 --rank 0
