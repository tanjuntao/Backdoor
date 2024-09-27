import numpy as np
import torch.optim.optimizer
from args_parser import get_args, get_model_dir, get_poison_epochs
from generate_poison import optimize_input
from surrogate_fn import finetune_surrogate
from termcolor import colored
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import Plain
from linkefl.dataio import MediaDataset, TorchDataset
from linkefl.messenger import EasySocket
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *  # noqa
from linkefl.util import num_input_nodes
from linkefl.vfl.nn import PassiveNeuralNetwork

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

    poison_epochs = get_poison_epochs()
    poison_epochs.insert(0, 1)  # insert epoch 1 at the beginning (index 0)
    poison_epochs.append(_epochs + 1)  # append total epochs at the end
    poison_samples = None
    for idx, curr_poison_epoch in enumerate(poison_epochs):
        # Load dataset
        if args.model in ("resnet18", "vgg13", "lenet"):
            passive_trainset = MediaDataset(
                role=Const.PASSIVE_NAME,
                dataset_name=args.dataset,
                root=_dataset_dir,
                train=True,
                download=True,
                poison_samples=poison_samples,
            )
            passive_testset = MediaDataset(
                role=Const.PASSIVE_NAME,
                dataset_name=args.dataset,
                root=_dataset_dir,
                train=False,
                download=True,
            )
            passive_validset = MediaDataset(
                role=Const.PASSIVE_NAME,
                dataset_name=args.dataset,
                root=_dataset_dir,
                train=True,
                download=True,
                valid=True,
            )
            active_validset = MediaDataset(
                role=Const.ACTIVE_NAME,
                dataset_name=args.dataset,
                root=_dataset_dir,
                train=True,
                download=True,
                valid=True,
            )
        elif args.model == "mlp":  # TODO: update poison dataset
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
        if curr_poison_epoch == 1:
            if args.model == "resnet18":
                bottom_model = ResNet18(in_channel=3, num_classes=num_classes).to(
                    _device
                )
            elif args.model == "vgg13":
                bottom_model = VGG13(in_channel=3, num_classes=num_classes).to(_device)
            elif args.model == "lenet":
                in_channel = 1
                if args.dataset == "svhn":
                    in_channel = 3
                bottom_model = LeNet(in_channel=in_channel, num_classes=num_classes).to(
                    _device
                )
            elif args.model == "mlp":
                input_nodes = num_input_nodes(
                    dataset_name=args.dataset,
                    role=Const.PASSIVE_NAME,
                    passive_feat_frac=_passive_feat_frac,
                )
                if args.dataset in ("tab_mnist", "tab_fashion_mnist"):
                    bottom_nodes = [input_nodes, 256, 128, 128]
                else:
                    bottom_nodes = [18, 15, 10, 10]
                bottom_model = MLP(
                    bottom_nodes,
                    activate_input=False,
                    activate_output=True,
                    random_state=_random_state,
                ).to(_device)
            else:
                raise ValueError(f"{args.model} is not an valid model type.")
            cut_layer = CutLayer(*_cut_nodes, random_state=_random_state).to(_device)
        else:
            pass_model = TorchModelIO.load(
                get_model_dir(), f"passive_epoch_{curr_poison_epoch-2}.model"
            )["model"]
            bottom_model = pass_model["bottom"].to(_device)
            cut_layer = pass_model["cut"].to(_device)
        print(bottom_model, cut_layer)
        _models = {"bottom": bottom_model, "cut": cut_layer}

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
                    f"{get_model_dir()}/optim/passive_optim_bottom_epoch_{curr_poison_epoch-2}.pth"
                )
            )
            _optimizers["cut"].load_state_dict(
                torch.load(
                    f"{get_model_dir()}/optim/passive_optim_cut_epoch_{curr_poison_epoch-2}.pth"
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
        passive_party = PassiveNeuralNetwork(
            epochs=poison_epochs[idx + 1] - curr_poison_epoch,
            start_epoch=curr_poison_epoch - 1,
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
            args=args,
        )
        passive_party.train(passive_trainset, passive_testset)
        print(colored("3. Passive party finish vfl_nn training.", "red"))

        # Terminate when there are no more poison epochs
        if poison_epochs[idx + 1] == _epochs + 1:
            break

        # Fine-tune surrogate model to obtain topk most confident target sample indices
        topk_indices = finetune_surrogate(poison_epochs[idx + 1] - 2, args)

        # Generate poison samples
        poison_samples = {}
        active_model = TorchModelIO.load(
            get_model_dir(), f"active_epoch_{poison_epochs[idx+1] - 2}.model"
        )["model"]
        passive_model = TorchModelIO.load(
            get_model_dir(), f"passive_epoch_{poison_epochs[idx+1] - 2}.model"
        )["model"]
        models = {  # TODO:
            "active_bottom": active_model["bottom"].to(_device),
            "passive_bottom": passive_model["bottom"].to(_device),
            "active_cut": active_model["cut"].to(_device),
            "passive_cut": passive_model["cut"].to(_device),
            "top": active_model["top"].to(_device),
        }
        np.random.shuffle(topk_indices)
        other_classes = list(range(num_classes))
        other_classes.remove(args.target)
        splitted_indices = np.array_split(topk_indices, len(other_classes))
        for idx, other in enumerate(other_classes):
            print(f"======> poisoning to class {other} <======")
            criterion = torch.nn.CrossEntropyLoss()
            chunk_indices = splitted_indices[idx].tolist()
            for sample_idx in chunk_indices:
                # print(active_validset[sample_idx][1])
                print(
                    active_validset[sample_idx][1], end=" "
                )  # print ground-truth label
                full_sample = {
                    "active": active_validset[sample_idx][0].to(
                        _device
                    ),  # obtain image
                    "passive": passive_validset[sample_idx][0].to(_device),
                }
                passive_poison_sample = optimize_input(
                    full_sample, models, criterion, args.target, other, agg=args.agg
                ).to("cpu")
                full_poison_sample = torch.cat(
                    (active_validset[sample_idx][0], passive_poison_sample), dim=1
                )
                poison_samples[sample_idx] = full_poison_sample
