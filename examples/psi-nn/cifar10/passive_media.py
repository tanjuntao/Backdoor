# from linkefl.modelzoo.security_model import resnet20
import argparse

import torch.optim.optimizer
from termcolor import colored
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory
from linkefl.dataio import MediaDataset
from linkefl.messenger import EasySocket
from linkefl.modelzoo import *  # noqa
from linkefl.vfl.nn import PassiveNeuralNetwork

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
    _crypto_type = Const.PLAIN
    _key_size = 1024
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    _cut_nodes = [10, 10]
    _random_state = None
    _saving_model = True
    _device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _messenger = EasySocket.init_passive(active_ip="localhost", active_port=args.port)
    _crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=100,
        gen_from_set=False,
    )
    _logger = logger_factory(role=Const.PASSIVE_NAME)

    # 1. Load dataset
    passive_trainset = MediaDataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root=_dataset_dir,
        train=True,
        download=True,
    )
    passive_testset = MediaDataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root=_dataset_dir,
        train=False,
        download=True,
    )
    print(colored("1. Finish loading dataset.", "red"))

    # 2. VFL training
    print(colored("2. Passive party started training...", "red"))
    bottom_model = ResNet18(in_channel=3, num_classes=10).to(_device)
    cut_layer = CutLayer(*_cut_nodes, random_state=_random_state).to(_device)

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

    # 3. Initialize vertical NN protocol and start fed training
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
        saving_model=_saving_model,
        random_state=_random_state,
        schedulers=schedulers,
        model_dir=f"/storage/1002tjt/MC-Attack/cifar10_mid_{args.mid_weight}",
        model_name="VFL_passive.model",
    )
    passive_party.train(passive_trainset, passive_testset)
    print(colored("3. Passive party finish vfl_nn training.", "red"))

    # 5. Close messenger, finish training
    _messenger.close()


### fedpass
# bottom_model = FedPassResNet18(
#     in_channel=3,
#     num_classes=10,
#     loc=-100,
#     passport_mode="multi",
#     scale=math.sqrt(args.sigma2),
# ).to(_device)
