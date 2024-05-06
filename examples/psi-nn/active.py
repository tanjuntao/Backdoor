import copy
import math

import torch
from args_parser import get_args, get_model_dir
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.messenger import EasySocketServer
from linkefl.modelzoo import *  # noqa: F405
from linkefl.util import num_input_nodes
from linkefl.vfl.nn import ActiveNeuralNetwork

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
        _top_nodes = [10, _n_classes]
    elif args.dataset == "cifar100":
        _batch_size = 128
        _dataset_dir = f"{data_prefix}/data"
        topk = 5
        _cut_nodes = [100, 100]
        _n_classes = 100
        _top_nodes = [100, _n_classes]
    elif args.dataset in ("mnist", "fashion_mnist", "svhn"):
        _batch_size = 128
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [10, 10]
        _n_classes = 10
        _top_nodes = [10, _n_classes]
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
    print(colored("2. Active party started training...", "red"))
    if args.model == "resnet18":
        bottom_model = ResNet18(in_channel=3, num_classes=_n_classes).to(_device)
    elif args.model == "vgg13":
        bottom_model = VGG13(in_channel=3, num_classes=_n_classes).to(_device)
    elif args.model == "lenet":
        in_channel = 1
        if args.dataset == "svhn":
            in_channel = 3
        bottom_model = LeNet(in_channel=in_channel, num_classes=_n_classes).to(_device)
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
    print(bottom_model, cut_layer, top_model)
    # Prepare FedPass Models
    if args.defense == "fedpass":
        if args.model == "mlp":
            # hidden, tm, tfm: 64, criteo: 128
            top_model = nn.Sequential(
                LinearPassportBlock(
                    in_features=64,
                    out_features=64,
                    # hidden_feature=128,
                    loc=-15,
                    passport_mode="multi",
                    scale=math.sqrt(args.sigma2),
                ),
                MLP(
                    _top_nodes,
                    activate_input=False,
                    activate_output=False,
                    random_state=_random_state,
                ),
            ).to(_device)
        if args.model == "lenet":
            if args.dataset == "svhn":
                in_channel = 3
            else:
                in_channel = 1
            bottom_model = FedPassLeNet(
                in_channel=in_channel,
                num_classes=10,
                loc=-100,
                scale=math.sqrt(args.sigma2),
            ).to(_device)
            top_model = nn.Sequential(
                LinearPassportBlock(
                    in_features=10,
                    out_features=10,
                    loc=-100,
                    passport_mode="multi",
                    scale=math.sqrt(args.sigma2),
                ),
                MLP(
                    _top_nodes,
                    activate_input=False,
                    activate_output=False,
                    random_state=_random_state,
                ),
            ).to(_device)
        if args.model == "vgg13":
            bottom_model = FedPassVGG13(
                in_channel=3,
                num_classes=100,
                loc=-100,
                scale=math.sqrt(args.sigma2),
            ).to(_device)
            top_model = nn.Sequential(
                LinearPassportBlock(
                    in_features=100,
                    out_features=100,
                    hidden_feature=1024,
                    loc=-100,
                    passport_mode="multi",
                    scale=math.sqrt(args.sigma2),
                ),
                MLP(
                    _top_nodes,
                    activate_input=False,
                    activate_output=False,
                    random_state=_random_state,
                ),
            ).to(_device)
        if args.model == "resnet18":
            bottom_model = FedPassResNet18(
                in_channel=3,
                num_classes=10,
                loc=-100,
                passport_mode="multi",
                scale=math.sqrt(args.sigma2),
            ).to(_device)
            top_model = nn.Sequential(
                LinearPassportBlock(
                    in_features=10,
                    out_features=10,
                    loc=-100,
                    passport_mode="multi",
                    scale=math.sqrt(args.sigma2),
                ),
                MLP(
                    _top_nodes,
                    activate_input=False,
                    activate_output=False,
                    random_state=_random_state,
                ),
            ).to(_device)
    print(bottom_model, cut_layer, top_model)
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
    # Init mid models
    # mid_model = DeepVIB(input_shape=64, output_shape=64, z_dim=64).to(_device)
    # mid_model = DeepVIB(input_shape=10, output_shape=10, z_dim=32).to(_device)
    mid_model = DeepVIB(input_shape=10, output_shape=10, z_dim=32).to(_device)
    mid_optimizer = torch.optim.SGD(
        mid_model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
    )
    mid_scheduler = CosineAnnealingLR(optimizer=mid_optimizer, T_max=_epochs, eta_min=0)

    # VMask
    if args.dataset == "criteo":
        bottom_nodes = [18, 15, 10, 10]
        bottom_model = MLP(
            bottom_nodes,
            activate_input=False,
            activate_output=True,
            random_state=_random_state,
        ).to(_device)
    shadow_model = nn.Sequential(copy.deepcopy(bottom_model), copy.deepcopy(cut_layer))
    shadow_optimizer = torch.optim.SGD(
        shadow_model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
    )
    shadow_scheduler = CosineAnnealingLR(
        optimizer=shadow_optimizer, T_max=_epochs, eta_min=0
    )
    mc_top_model = copy.deepcopy(top_model)
    if args.model in ("resnet18", "vgg13", "lenet"):
        shadow_dataset = MediaDataset(
            role=Const.PASSIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=True,
            download=True,
            fine_tune=True,
            fine_tune_per_class=args.shadow_per_class,
        )
        mc_dataset = MediaDataset(
            role=Const.PASSIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=True,
            download=True,
            fine_tune=True,
            fine_tune_per_class=args.per_class,
        )
        passive_dataset = MediaDataset(
            role=Const.PASSIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=False,
            download=True,
        )
    elif args.model == "mlp":
        _passive_feat_frac = 0.5
        _feat_perm_option = Const.SEQUENCE
        shadow_dataset = TorchDataset.buildin_dataset(
            dataset_name=args.dataset,
            role=Const.PASSIVE_NAME,
            root=_dataset_dir,
            train=True,
            download=True,
            passive_feat_frac=_passive_feat_frac,
            feat_perm_option=_feat_perm_option,
            seed=_random_state,
            fine_tune=True,
            fine_tune_per_class=args.shadow_per_class,
        )
        mc_dataset = TorchDataset.buildin_dataset(
            dataset_name=args.dataset,
            role=Const.PASSIVE_NAME,
            root=_dataset_dir,
            train=True,
            download=True,
            passive_feat_frac=_passive_feat_frac,
            feat_perm_option=_feat_perm_option,
            seed=_random_state,
            fine_tune=True,
            fine_tune_per_class=args.per_class,
        )
        passive_dataset = TorchDataset.buildin_dataset(
            dataset_name=args.dataset,
            role=Const.PASSIVE_NAME,
            root=_dataset_dir,
            train=False,
            download=True,
            passive_feat_frac=_passive_feat_frac,
            feat_perm_option=_feat_perm_option,
            seed=_random_state,
        )
    else:
        raise ValueError(f"{args.model} is not an valid model type.")

    # Model training
    active_party = ActiveNeuralNetwork(
        epochs=_epochs,
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
        defense=args.defense if args.defense else None,
        ng_noise_scale=args.noise_scale,
        cg_preserved_ratio=args.preserved_ratio,
        dg_bins_num=args.bins_num,
        labeldp_eps=args.eps,
        dcor_weight=args.dcor_weight,
        mid_weight=args.mid_weight,
        mid_model=mid_model,
        mid_optimizer=mid_optimizer,
        mid_scheduler=mid_scheduler,
        shadow_model=shadow_model,
        shadow_optimizer=shadow_optimizer,
        shadow_scheduler=shadow_scheduler,
        shadow_dataset=shadow_dataset,
        mc_dataset=mc_dataset,
        mc_top_model=mc_top_model,
        passive_dataset=passive_dataset,
        privacy_budget=args.privacy_budget,
        args=args,
    )
    active_party.train(active_trainset, active_testset)
    print(colored("3. Active party finished vfl_nn training.", "red"))
    for messenger in _messengers:
        messenger.close()
