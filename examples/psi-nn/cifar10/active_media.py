import argparse

import torch
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset
from linkefl.messenger import EasySocketServer
from linkefl.modelzoo import *  # noqa
from linkefl.vfl.nn import ActiveNeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int, help="gpu device")
parser.add_argument("--port", default=20000, type=int, help="active port")
parser.add_argument("--noise_scale", default=1e-5, type=float, help="ng noise scale")
parser.add_argument(
    "--preserved_ratio", default=0.1, type=float, help="cg preserved ratio"
)
parser.add_argument("--bins_num", default=100, type=int, help="dg bins num")
parser.add_argument("--eps", default=1.0, type=float, help="labeldp epsilon")
parser.add_argument(
    "--sigma2", default=1.0, type=float, help="fedpass secret key scale square"
)
parser.add_argument(
    "--mid_weight", default=1.0, type=float, help="weight of vib loss in mid"
)
args = parser.parse_args()

# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # 0. Set parameters
    _dataset_dir = "../data"
    _dataset_name = "cifar10"
    _epochs = 50
    _batch_size = 128
    _learning_rate = 0.1
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = None
    _device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _saving_model = True
    _cut_nodes = [10, 10]
    _n_classes = 10
    _top_nodes = [10, _n_classes]
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _messengers = EasySocketServer(
        active_ip="localhost", active_port=args.port, passive_num=1
    ).get_messengers()

    # 1. Load dataset
    active_trainset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root=_dataset_dir,
        train=True,
        download=True,
    )
    active_testset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root=_dataset_dir,
        train=False,
        download=True,
    )
    print(colored("1. Finish loading dataset.", "red"))

    # 2. VFL training
    print(colored("2. Active party started training...", "red"))
    bottom_model = ResNet18(in_channel=3, num_classes=10).to(_device)
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
    schedulers = {
        name: CosineAnnealingLR(optimizer=optimizer, T_max=_epochs, eta_min=0)
        for name, optimizer in _optimizers.items()
    }

    # Initialize vertical NN protocol and start training
    mid_model = DeepVIB(input_shape=10, output_shape=10, z_dim=32).to(_device)
    mid_optimizer = torch.optim.SGD(
        mid_model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
    )
    mid_scheduler = CosineAnnealingLR(optimizer=mid_optimizer, T_max=_epochs, eta_min=0)
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
        random_state=_random_state,
        saving_model=_saving_model,
        schedulers=schedulers,
        model_dir=f"/storage/1002tjt/MC-Attack/cifar10_mid_{args.mid_weight}",
        model_name="VFL_active.model",
        save_every_epoch=True,
        defense="mid",
        mid_weight=args.mid_weight,
        mid_model=mid_model,
        mid_optimizer=mid_optimizer,
        mid_scheduler=mid_scheduler,
        # labeldp_eps=args.eps,
        # dg_bins_num=args.bins_num,
        # cg_preserved_ratio=args.preserved_ratio,
        # ng_noise_scale=args.noise_scale,
        # defense="labeldp",
        # eps=2,
    )
    active_party.train(active_trainset, active_testset)
    print(colored("3. Active party finished vfl_nn training.", "red"))

    # 3. Close messenger, finish training
    for messenger in _messengers:
        messenger.close()


### FedPass bottom model and top model ###
# bottom_model = FedPassResNet18(
#     in_channel=3,
#     num_classes=10,
#     loc=-100,
#     passport_mode="multi",
#     scale=math.sqrt(args.sigma2),
# ).to(_device)
# top_model = nn.Sequential(
#     LinearPassportBlock(
#         in_features=10,
#         out_features=10,
#         loc=-100,
#         passport_mode="multi",
#         scale=math.sqrt(args.sigma2),
#     ),
#     MLP(_top_nodes, activate_input=False, activate_output=False, random_state=_random_state),
# ).to(_device)


### MID bottom model
