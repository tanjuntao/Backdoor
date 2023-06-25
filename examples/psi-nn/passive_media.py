import torch.optim.optimizer
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory
from linkefl.dataio import MediaDataset
from linkefl.messenger import FastSocket
from linkefl.modelzoo import *
from linkefl.vfl.nn import PassiveNeuralNetwork
from linkefl.modelzoo.security_model import resnet20

# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # 0. Set parameters
    _dataset_dir = "data"
    _dataset_name = "cifar10"
    _active_ip = "localhost"
    _active_port = 20000
    _passive_ip = "localhost"
    _passive_port = 30000
    _epochs = 50
    _batch_size = 128
    _learning_rate = 0.1
    _crypto_type = Const.PLAIN
    _key_size = 1024
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    _cut_nodes = [10, 10]
    _random_state = None
    _saving_model = True
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=_active_ip,
        active_port=_active_port,
        passive_ip=_passive_ip,
        passive_port=_passive_port,
    )
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
    bottom_model = ResNet18(in_channel=3).to(_device)
    # bottom_model = resnet20().to(_device)
    cut_layer = CutLayer(*_cut_nodes, random_state=_random_state).to(_device)
    _models = {"bottom": bottom_model, "cut": cut_layer}
    _optimizers = {
        name: torch.optim.SGD(
            model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
        )
        for name, model in _models.items()
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
    )
    passive_party.train(passive_trainset, passive_testset)
    print(colored("3. Passive party finish vfl_nn training.", "red"))

    # 5. Close messenger, finish training
    _messenger.close()
