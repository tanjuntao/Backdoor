import torch
from args_parser import get_args, get_model_dir
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *
from linkefl.util import num_input_nodes
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
    iterate_epochs = 50
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
        passive_validset = MediaDataset(
            role=Const.PASSIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=True,
            download=True,
            valid=True,
        )
        passive_testset = MediaDataset(
            role=Const.PASSIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=False,
            download=True,
        )
        fine_tune_trainset = MediaDataset(
            role=Const.PASSIVE_NAME,
            dataset_name=args.dataset,
            root=_dataset_dir,
            train=True,
            download=True,
            fine_tune=True,
            fine_tune_per_class=args.per_class,
        )
    elif args.model == "mlp":  # TODO: add passive validset
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
        fine_tune_trainset = TorchDataset.buildin_dataset(
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
    # print(fine_tune_trainset.buildin_dataset.data.shape)
    # print(len(fine_tune_trainset.buildin_dataset.targets))
    # print(fine_tune_trainset.buildin_dataset.targets)
    print(colored("1. Finish loading dataset.", "red"))

    epoch_surrogate_accs = []
    epoch_topk_confident_accs = []
    for epoch in range(iterate_epochs):
        # Load model
        if iterate_epochs == 1:
            bottom_model = TorchModelIO.load(get_model_dir(), "VFL_passive.model")[
                "model"
            ]["bottom"].to(_device)
        else:
            bottom_model = TorchModelIO.load(
                get_model_dir(), f"passive_epoch_{epoch}.model"
            )["model"]["bottom"].to(_device)

        if args.scratch:
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
                    bottom_nodes = [18, 15, 10]
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
        _models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}
        # for param in bottom_model.parameters():
        #     param.requires_grad = False
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
        # del _optimizers["bottom"]

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
            args=args,
        )
        surrogate_acc = active_party.train_alone(
            fine_tune_trainset, passive_testset
        )  # return value is a tuple
        scores, _ = active_party.validate_alone(
            passive_validset, cal_topk_acc=True, topk_confident=args.topk_confident
        )
        print(colored("3. Active party finished vfl_nn training.", "red"))

        epoch_surrogate_accs.append(surrogate_acc)
        print(
            f"=======================> epoch: {epoch}, surrogate_acc: {surrogate_acc}"
        )
        epoch_topk_confident_accs.append(scores["topk_confident_acc"])
        print(
            f"=======================> epoch: {epoch}, topk_confident_acc:"
            f" {scores['topk_confident_acc']}"
        )

        print("=========> final surrogate accs")
        size = len(epoch_surrogate_accs[0])
        for i in range(size):
            accs = [t[i] for t in epoch_surrogate_accs]
            print(accs)

        print("=========> final topk confident accs")
        print(epoch_topk_confident_accs)
