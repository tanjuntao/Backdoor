import argparse

import torch
from termcolor import colored
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset
from linkefl.modelio import TorchModelIO
from linkefl.modelzoo import *
from linkefl.vfl.nn import ActiveNeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int, help="gpu device")
args = parser.parse_args()

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


def get_layer(index, model, uniform_random=False):
    pass


# fmt: off
if __name__ == "__main__":
    # 0. Set parameters
    _dataset_dir = "../data"
    _dataset_name = "cifar10"
    _epochs = 50
    _batch_size = 8
    _learning_rate = 0.01
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = None
    _device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _cut_nodes = [10, 10]
    _n_classes = 10
    _top_nodes = [10, _n_classes]
    _logger = logger_factory(role=Const.ACTIVE_NAME)

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
    fine_tune_trainset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root=_dataset_dir,
        train=True,
        download=True,
        fine_tune=True,
        fine_tune_per_class=32,
    )

    print(fine_tune_trainset.buildin_dataset.data.shape)
    print(len(fine_tune_trainset.buildin_dataset.targets))
    print(fine_tune_trainset.buildin_dataset.targets)
    print(colored("1. Finish loading dataset.", "red"))

    # 2. VFL training
    print(colored("2. Active party started training...", "red"))
    # bottom_model = ResNet18(in_channel=3, num_classes=10).to(_device)
    # init_model(bottom_model, range=10000000)
    bottom_model = TorchModelIO.load("../models/cifar10", "VFL_active.model")["model"][
        "bottom"
    ].to(_device)
    # bottom_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(_device)
    # bottom_model.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False).to(_device)
    # bottom_model.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer1[1].conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer1[1].conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer2[1].conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer2[1].conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer3[1].conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer3[1].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to(_device)
    # bottom_model.linear = nn.Linear(in_features=512, out_features=10, bias=True).to(_device)

    # bottom_model.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)
    # bottom_model.layer1[0].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)
    # bottom_model.layer1[0].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)
    # bottom_model.layer1[1].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)
    # bottom_model.layer1[1].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(_device)

    # bottom_model.layer1[0].conv1.weight.data[:bottom_model.layer1[0].conv1.weight.shape[0]//2] = torch.rand(bottom_model.layer1[0].conv1.weight.shape[0]//2, bottom_model.layer1[0].conv1.weight.shape[1], 3, 3) * 2 - 1
    # bottom_model.layer1[0].conv2.weight.data[:bottom_model.layer1[0].conv2.weight.shape[0]//2] = torch.rand(bottom_model.layer1[0].conv2.weight.shape[0]//2, bottom_model.layer1[0].conv2.weight.shape[1], 3, 3) * 2 - 1
    # bottom_model.layer1[1].conv1.weight.data[:bottom_model.layer1[1].conv1.weight.shape[0]//2] = torch.rand(bottom_model.layer1[1].conv1.weight.shape[0]//2, bottom_model.layer1[1].conv1.weight.shape[1], 3, 3) * 2 - 1
    # bottom_model.layer1[1].conv2.weight.data[:bottom_model.layer1[1].conv2.weight.shape[0]//2] = torch.rand(bottom_model.layer1[1].conv2.weight.shape[0]//2, bottom_model.layer1[1].conv2.weight.shape[1], 3, 3) * 2 - 1

    # bottom_model.conv1.apply(init_uniform)
    # bottom_model.5ayer1[0].conv1.apply(init_uniform)
    # bottom_model.layer1[0].conv2.apply(init_uniform)
    # bottom_model.layer1[1].conv1.apply(init_uniform)
    # bottom_model.layer1[1].conv2.apply(init_uniform)
    # bottom_model.layer2[0].conv1.apply(init_uniform)
    # bottom_model.layer2[0].conv2.apply(init_uniform)
    # bottom_model.layer2[1].conv1.apply(init_uniform)
    # bottom_model.layer2[1].conv2.apply(init_uniform)
    # bottom_model.layer3[0].conv1.apply(init_uniform)
    # bottom_model.layer3[0].conv2.apply(init_uniform)
    # bottom_model.layer3[1].conv1.apply(init_uniform)
    # bottom_model.layer3[1].conv2.apply(init_uniform)
    # bottom_model.layer4[0].conv1.apply(init_uniform)
    # bottom_model.layer4[0].conv2.apply(init_uniform)
    # bottom_model.layer4[1].conv1.apply(init_uniform)
    # bottom_model.layer4[1].conv2.apply(init_uniform)
    # bottom_model.linear.apply(init_uniform)

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
    # Initialize vertical NN protocol and start training
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
        random_state=_random_state,
        saving_model=False,
        schedulers=schedulers,
        model_dir="../models/cifar10",
        model_name="attack.model",
    )
    active_party.validate_alone(active_testset)
    active_party.train_alone(fine_tune_trainset, active_testset)

    # bottom_model = TorchModelIO.load("models/cifar1030", "fine_tune_active.model")["model"]["bottom"].to(_device)
    # cut_layer = TorchModelIO.load("models/cifar1030", "fine_tune_active.model")["model"]["cut"].to(_device)
    # top_model = TorchModelIO.load("models/cifar1030", "fine_tune_active.model")["model"]["top"].to(_device)
    # _models = {"bottom": bottom_model, "cut": cut_layer, "top": top_model}
    # active_party.models = _models
    active_party.validate_alone(active_testset)
    # _, total_embeddings = active_party.validate_alone(active_testset)
    print(colored("3. Active party finished vfl_nn training.", "red"))

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

# fmt: off
