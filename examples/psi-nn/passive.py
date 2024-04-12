import math

import torch.optim.optimizer
from args_parser import get_args, get_model_dir
from termcolor import colored
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import Plain
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.messenger import EasySocket
from linkefl.modelzoo import *  # noqa
from linkefl.util import num_input_nodes
from linkefl.vfl.nn import PassiveNeuralNetwork

args = get_args()

# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Set params
    data_prefix = "."
    _epochs = 50
    _learning_rate = 0.1
    _random_state = None
    _device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _messenger = EasySocket.init_passive(active_ip="localhost", active_port=args.port)
    _crypto = Plain()
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    if args.dataset in ("cifar10", "cinic10"):
        if args.dataset == "cifar10":
            _batch_size = 128
            _dataset_dir = f"{data_prefix}/data"
        else:
            _batch_size = 256
            _dataset_dir = f"{data_prefix}/data/CINIC10"
        num_classes = 10
        _cut_nodes = [10, 10]
    elif args.dataset == "cifar100":
        _batch_size = 128
        _dataset_dir = f"{data_prefix}/data"
        num_classes = 100
        _cut_nodes = [100, 100]
    elif args.dataset in ("mnist", "fashion_mnist", "svhn"):
        _batch_size = 128
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [10, 10]
        num_classes = 10
    elif args.dataset in ("tab_mnist", "tab_fashion_mnist"):
        _batch_size = 128
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [128, 64]
        num_classes = 10
    elif args.dataset == "criteo":
        _batch_size = 256
        _dataset_dir = f"{data_prefix}/data"
        topk = 1
        _cut_nodes = [10, 10]
        num_classes = 2
    else:
        raise ValueError(f"{args.dataset} is not valid dataset.")

    # Load dataset
    if args.model in ("resnet18", "vgg13", "lenet"):
        passive_trainset = MediaDataset(
            role=Const.PASSIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=True,
            download=True,
        )
        passive_testset = MediaDataset(
            role=Const.PASSIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=False,
            download=True,
        )
    elif args.model == "mlp":
        _passive_feat_frac = 0.5
        _feat_perm_option = Const.SEQUENCE
        passive_trainset = TorchDataset.buildin_dataset(
            dataset_name=args.dataset,
            role=Const.PASSIVE_NAME,
            root=_dataset_dir,
            train=True,
            download=True,
            passive_feat_frac=_passive_feat_frac,
            feat_perm_option=_feat_perm_option,
            seed=_random_state,
        )
        passive_testset = TorchDataset.buildin_dataset(
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

    print(colored("1. Finish loading dataset.", "red"))

    # Init models
    print(colored("2. Passive party started training...", "red"))
    if args.model == "resnet18":
        bottom_model = ResNet18(in_channel=3, num_classes=num_classes).to(_device)
    elif args.model == "vgg13":
        bottom_model = VGG13(in_channel=3, num_classes=num_classes).to(_device)
    elif args.model == "lenet":
        in_channel = 1
        if args.dataset == "svhn":
            in_channel = 3
        bottom_model = LeNet(in_channel=in_channel, num_classes=num_classes).to(_device)
    elif args.model == "mlp":
        input_nodes = num_input_nodes(
            dataset_name=args.dataset,
            role=Const.PASSIVE_NAME,
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

    # Prepare FedPass Models
    if args.defense == "fedpass":
        if args.model == "mlp":
            pass
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
        if args.model == "vgg13":
            bottom_model = FedPassVGG13(
                in_channel=3,
                num_classes=100,
                loc=-100,
                scale=math.sqrt(args.sigma2),
            ).to(_device)
        if args.model == "resnet18":
            bottom_model = FedPassResNet18(
                in_channel=3,
                num_classes=100,
                loc=-100,
                passport_mode="multi",
                scale=math.sqrt(args.sigma2),
            ).to(_device)

    _models = {"bottom": bottom_model, "cut": cut_layer}
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

    # Model training
    passive_party = PassiveNeuralNetwork(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=_models,
        optimizers=_optimizers,
        messenger=_messenger,
        cryptosystem=_crypto,
        logger=_logger,
        device=_device,
        num_workers=1,
        val_freq=1,
        saving_model=True,
        random_state=_random_state,
        schedulers=schedulers,
        model_dir=get_model_dir(),
        model_name="VFL_passive.model",
    )
    passive_party.train(passive_trainset, passive_testset)
    print(colored("3. Passive party finish vfl_nn training.", "red"))
    _messenger.close()


# fedpass
# bottom_model = FedPassResNet18(
#     in_channel=3,
#     num_classes=10,
#     loc=-100,
#     passport_mode="multi",
#     scale=math.sqrt(args.sigma2),
# ).to(_device)
