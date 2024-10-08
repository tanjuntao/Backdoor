import torch
from args_parser import get_args, get_model_dir
from termcolor import colored
from torch import nn

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.messenger.easysocket import EasySocketServer
from linkefl.modelio import TorchModelIO
from linkefl.vfl.nn import ActiveNeuralNetwork

args = get_args()

# seed = 3
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def init_uniform(module):
    nn.init.uniform_(module.weight, -10000000, 10000000)


def init_model(model, range):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight, -range, range)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, -range, range)
            nn.init.uniform_(m.bias, -range, range)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -range, range)
            nn.init.uniform_(m.bias, -range, range)


if __name__ == "__main__":
    # Set params
    data_prefix = "."
    _epochs = 50
    _learning_rate = 0.01
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
            _dataset_dir = f"{data_prefix}/data"
        else:
            _dataset_dir = f"{data_prefix}/data/CINIC10"
        topk = 1
        _batch_size = 128
        _cut_nodes = [10, 10]
        _n_classes = 10
        _top_nodes = [10, _n_classes]
    elif args.dataset == "cifar100":
        _dataset_dir = f"{data_prefix}/data"
        topk = 5
        _batch_size = 256
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
    # print(fine_tune_trainset.buildin_dataset.data.shape)
    # print(len(fine_tune_trainset.buildin_dataset.targets))
    # print(fine_tune_trainset.buildin_dataset.targets)
    print(colored("1. Finish loading dataset.", "red"))

    # Load model
    bottom_model = TorchModelIO.load(get_model_dir(), "VFL_active.model")["model"][
        "bottom"
    ].to(_device)
    cut_layer = TorchModelIO.load(get_model_dir(), "VFL_active.model")["model"][
        "cut"
    ].to(_device)
    top_model = TorchModelIO.load(get_model_dir(), "VFL_active.model")["model"][
        "top"
    ].to(_device)

    _models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}

    active_party = ActiveNeuralNetwork(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=_models,
        optimizers=None,
        loss_fn=_loss_fn,
        messengers=_messengers,
        logger=_logger,
        device=_device,
        num_workers=1,
        val_freq=1,
        topk=topk,
        random_state=_random_state,
        saving_model=False,
        schedulers=None,
        args=args,
    )
    active_party.validate_attack(active_testset)
    print(colored("3. Active party finished vfl_nn training.", "red"))
