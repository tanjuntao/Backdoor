import torch
from args_parser import get_args, get_mask_layers, get_model_dir
from mask import layer_masking
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *
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
    _epochs = 50
    _learning_rate = 0.01
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = None
    _device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    if args.dataset in ("cifar10", "cinic10"):
        if args.dataset == "cifar10":
            _dataset_dir = "../data"
        else:
            _dataset_dir = "../data/CINIC10"
        topk = 1
        _batch_size = 4
        _cut_nodes = [10, 10]
        _n_classes = 10
        _top_nodes = [10, _n_classes]
    elif args.dataset == "cifar100":
        _dataset_dir = "../data"
        topk = 5
        _batch_size = 8
        _cut_nodes = [100, 100]
        _n_classes = 100
        _top_nodes = [100, _n_classes]
    elif args.dataset in ("mnist", "fashion_mnist", "svhn"):
        _batch_size = 4
        _dataset_dir = "../data"
        topk = 1
        _cut_nodes = [10, 10]
        _n_classes = 10
        _top_nodes = [10, _n_classes]
    else:
        raise ValueError(f"{args.dataset} is not valid dataset.")

    # Load dataset
    active_testset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name=args.dataset,
        root=_dataset_dir,
        train=False,
        download=True,
    )
    fine_tune_trainset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name=args.dataset,
        root=_dataset_dir,
        train=True,
        download=True,
        fine_tune=True,
        fine_tune_per_class=args.per_class,
    )
    # print(fine_tune_trainset.buildin_dataset.data.shape)
    # print(len(fine_tune_trainset.buildin_dataset.targets))
    # print(fine_tune_trainset.buildin_dataset.targets)
    print(colored("1. Finish loading dataset.", "red"))

    # Load model
    bottom_model = TorchModelIO.load(get_model_dir(), "VFL_active.model")["model"][
        "bottom"
    ].to(_device)
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
        else:
            raise ValueError(f"{args.model} is not an valid model type.")
    bottom_model = layer_masking(
        model_type=args.model,
        bottom_model=bottom_model,
        mask_layers=get_mask_layers(),
        device=_device,
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

    # fmt: off
    # Visualization
    # active_party.validate_alone(active_testset)
    # bottom_model = TorchModelIO.load(
    #     "models/cifar1030", "fine_tune_active.model")["model"]["bottom"].to(_device)
    # cut_layer = TorchModelIO.load(
    #     "models/cifar1030", "fine_tune_active.model")["model"]["cut"].to(_device)
    # top_model = TorchModelIO.load(
    #     "models/cifar1030", "fine_tune_active.model")["model"]["top"].to(_device)
    # _models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}
    # active_party.models = _models
    # active_party.validate_alone(active_testset)
    # _, total_embeddings = active_party.validate_alone(active_testset)

    # # tsne visualization
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # import numpy as np
    # tsne = TSNE(n_components=2, random_state=0)
    # transformed_embeddings = tsne.fit_transform(total_embeddings.cpu().numpy())
    # test_targets = np.array(active_testset.buildin_dataset.targets)
    # for i in range(10):
    #     plt.scatter(
    #         transformed_embeddings[test_targets == i, 0],
    #         transformed_embeddings[test_targets == i, 1],
    #         label=str(i),
    #     )
    # plt.legend()
    # plt.savefig("models/cifar10_exp/embeddings_scratch.png")
    # fmt: on
