import pickle

import torch
from args_parser import get_args, get_model_dir
from termcolor import colored
from torch import nn

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import Plain
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.messenger import EasySocket
from linkefl.modelio import TorchModelIO
from linkefl.vfl.nn import PassiveNeuralNetwork

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
    _messenger = EasySocket.init_passive(active_ip="localhost", active_port=args.port)
    _random_state = None
    _device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _crypto = Plain()
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
    print(colored("1. Finish loading dataset.", "red"))

    # Load model
    bottom_model = TorchModelIO.load(get_model_dir(), "VFL_passive.model")["model"][
        "bottom"
    ].to(_device)
    cut_layer = TorchModelIO.load(get_model_dir(), "VFL_passive.model")["model"][
        "cut"
    ].to(_device)
    models = {"bottom": bottom_model, "cut": cut_layer}

    # Model training
    passive_party = PassiveNeuralNetwork(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        models=models,
        optimizers=None,
        messenger=_messenger,
        cryptosystem=_crypto,
        logger=_logger,
        device=_device,
        num_workers=1,
        val_freq=1,
        random_state=_random_state,
        saving_model=False,
        schedulers=None,
        args=args,
    )

    with open(
        f"{get_model_dir()}/target_embedding_mean_class_{args.target}.np", "rb"
    ) as f:
        trigger_embedding = pickle.load(f)
        print(trigger_embedding)
    trigger_embedding = torch.from_numpy(trigger_embedding).to(_device)
    # passive_party.validate_attack(passive_testset)
    passive_party.validate_attack(passive_testset, trigger_embedding=trigger_embedding)
    print(colored("3. passive party finished vfl_nn training.", "red"))
