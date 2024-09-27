import torch
from args_parser import get_args, get_model_dir, get_poison_epochs
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.messenger import EasySocketServer
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *  # noqa: F405
from linkefl.util import num_input_nodes
from linkefl.vfl.nn import ActiveNeuralNetwork
from linkefl.vfl.utils.evaluate import Plot

args = get_args()

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Set params
    data_prefix = "."
    _epochs = 50
    _learning_rate = 0.1
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = None
    _device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _messengers = EasySocketServer(
        active_ip="localhost",
        active_port=args.port,
        passive_num=1,
    ).get_messengers()
    if args.dataset in ("cifar10", "cinic10"):
        if args.dataset == "cifar10":
            _batch_size = 128
            _dataset_dir = f"{data_prefix}/data"
        else:
            _batch_size = 256
            _dataset_dir = f"{data_prefix}/data/CINIC10"
        topk = 1
        _cut_nodes = [10, 10]
        _n_classes = 10
        if args.agg == "add":
            _top_nodes = [10, _n_classes]
        elif args.agg == "concat":
            _top_nodes = [20, 20, _n_classes]
        else:
            raise ValueError(f"invalid aggregator: {args.agg}")
    elif args.dataset == "cifar100":
        _batch_size = 128
        _dataset_dir = f"{data_prefix}/data"
        topk = 5
        _cut_nodes = [100, 100]
        _n_classes = 100
        if args.agg == "add":
            _top_nodes = [100, _n_classes]
        elif args.agg == "concat":
            _top_nodes = [200, 200, _n_classes]
        else:
            raise ValueError(f"invalid aggregator: {args.agg}")
    elif args.dataset in ("mnist", "fashion_mnist", "svhn"):
        _batch_size = 128
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [10, 10]
        _n_classes = 10
        if args.agg == "add":
            _top_nodes = [10, _n_classes]
        elif args.agg == "concat":
            _top_nodes = [20, 20, _n_classes]
    elif args.dataset in ("tab_mnist", "tab_fashion_mnist"):
        _batch_size = 128
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [128, 64]
        _n_classes = 10
        _top_nodes = [64, _n_classes]
    elif args.dataset == "criteo":
        _batch_size = 256
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [10, 10]
        _n_classes = 2
        _top_nodes = [10, _n_classes]
    else:
        raise ValueError(f"{args.dataset} is not valid dataset.")

    poison_epochs = get_poison_epochs()
    poison_epochs.insert(0, 1)  # insert epoch 1 at the beginning (index 0)
    poison_epochs.append(_epochs + 1)  # append total epochs at the end
    best_acc = 0
    train_loss_records, valid_loss_records, train_acc_records, valid_acc_records = (
        [],
        [],
        [],
        [],
    )
    for idx, curr_poison_epoch in enumerate(poison_epochs):
        # Load dataset
        if args.model in ("resnet18", "vgg13", "lenet"):
            active_trainset = MediaDataset(
                role=Const.ACTIVE_NAME,
                dataset_name=args.dataset,
                root=_dataset_dir,
                train=True,
                download=True,
            )
            active_testset = MediaDataset(
                role=Const.ACTIVE_NAME,
                dataset_name=args.dataset,
                root=_dataset_dir,
                train=False,
                download=True,
            )
        elif args.model == "mlp":
            _passive_feat_frac = 0.5
            _feat_perm_option = Const.SEQUENCE
            active_trainset = TorchDataset.buildin_dataset(
                dataset_name=args.dataset,
                role=Const.ACTIVE_NAME,
                root=_dataset_dir,
                train=True,
                download=True,
                passive_feat_frac=_passive_feat_frac,
                feat_perm_option=_feat_perm_option,
                seed=_random_state,
            )
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
        else:
            raise ValueError(f"{args.model} is not an valid model type.")
        print(colored("1. Finish loading dataset.", "red"))

        # Init models
        if curr_poison_epoch == 1:
            if args.model == "resnet18":
                bottom_model = ResNet18(in_channel=3, num_classes=_n_classes).to(
                    _device
                )
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
                input_nodes = num_input_nodes(
                    dataset_name=args.dataset,
                    role=Const.ACTIVE_NAME,
                    passive_feat_frac=_passive_feat_frac,
                )
                if args.dataset in ("tab_mnist", "tab_fashion_mnist"):
                    bottom_nodes = [input_nodes, 256, 128, 128]
                else:
                    bottom_nodes = [input_nodes, 15, 10, 10]
                bottom_model = MLP(
                    bottom_nodes,
                    activate_input=False,
                    activate_output=True,
                    random_state=_random_state,
                ).to(_device)
            else:
                raise ValueError(f"{args.model} is not an valid model type.")
            cut_layer = CutLayer(*_cut_nodes, random_state=_random_state).to(_device)
            top_model = MLP(
                _top_nodes,
                activate_input=True,
                activate_output=False,
                random_state=_random_state,
            ).to(_device)
        else:
            acti_model = TorchModelIO.load(
                get_model_dir(), f"active_epoch_{curr_poison_epoch-2}.model"
            )["model"]
            bottom_model = acti_model["bottom"].to(_device)
            cut_layer = acti_model["cut"].to(_device)
            top_model = acti_model["top"].to(_device)
        print(bottom_model, cut_layer, top_model)
        _models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}

        # Init optimizers
        _optimizers = {
            name: torch.optim.SGD(
                model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
            )
            for name, model in _models.items()
        }
        if curr_poison_epoch == 1:
            pass
        else:
            _optimizers["bottom"].load_state_dict(
                torch.load(
                    f"{get_model_dir()}/optim/active_optim_bottom_epoch_{curr_poison_epoch-2}.pth"
                )
            )
            _optimizers["cut"].load_state_dict(
                torch.load(
                    f"{get_model_dir()}/optim/active_optim_cut_epoch_{curr_poison_epoch-2}.pth"
                )
            )
            _optimizers["top"].load_state_dict(
                torch.load(
                    f"{get_model_dir()}/optim/active_optim_top_epoch_{curr_poison_epoch-2}.pth"
                )
            )

        # Init schedulers
        last_epoch = -1 if curr_poison_epoch == 1 else curr_poison_epoch - 2
        schedulers = {
            name: CosineAnnealingLR(
                optimizer=optimizer, T_max=_epochs, eta_min=0, last_epoch=last_epoch
            )
            for name, optimizer in _optimizers.items()
        }

        # Model training
        weight = torch.tensor([1.0] * _n_classes).to(_device)
        weight[args.target] = weight[args.target] * args.weight_scale
        _loss_fn = nn.CrossEntropyLoss(weight=weight)
        active_party = ActiveNeuralNetwork(
            epochs=poison_epochs[idx + 1] - curr_poison_epoch,
            start_epoch=curr_poison_epoch - 1,
            batch_size=_batch_size,
            learning_rate=_learning_rate,
            models=_models,
            optimizers=_optimizers,
            loss_fn=_loss_fn,
            messengers=_messengers,
            logger=_logger,
            device=_device,
            num_workers=1,
            val_freq=1,
            topk=topk,
            random_state=_random_state,
            saving_model=True,
            schedulers=schedulers,
            model_dir=get_model_dir(),
            model_name="VFL_active.model",
            save_every_epoch=True,
            args=args,
            best_acc=best_acc,
        )
        scores = active_party.train(active_trainset, active_testset)
        best_acc = scores["best_acc"]
        train_loss_records.extend(scores["train_loss_records"])
        valid_loss_records.extend(scores["valid_loss_records"])
        train_acc_records.extend(scores["train_acc_records"])
        valid_acc_records.extend(scores["valid_acc_records"])
        print(colored("3. Active party finished vfl_nn training.", "red"))

        # Terminate when there are no more poison epochs
        if poison_epochs[idx + 1] == _epochs + 1:
            break

    Plot.plot_train_test_loss(train_loss_records, valid_loss_records, get_model_dir())
    Plot.plot_train_test_acc(train_acc_records, valid_acc_records, get_model_dir())
