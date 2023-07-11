import torch.optim.optimizer
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory
from linkefl.dataio import TorchDataset
from linkefl.messenger import FastSocket
from linkefl.modelzoo import MLP, CutLayer
from linkefl.psi import PassiveCM20PSI
from linkefl.vfl.nn import PassiveNeuralNetwork

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_path = "./data/tabmnist-passive001.csv"
    _has_header = False
    _test_size = 0.2
    _active_ip = "localhost"
    _active_port = 20000
    _passive_ip = "localhost"
    _passive_port = 30000
    _epochs = 50
    _batch_size = 256
    _learning_rate = 0.01
    _crypto_type = Const.PLAIN
    _key_size = 1024
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    _bottom_nodes = [392, 256, 128]
    _cut_nodes = [128, 64]
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
    passive_dataset = TorchDataset.from_csv(
        role=Const.PASSIVE_NAME,
        abs_path=_dataset_path,
        dataset_type=Const.CLASSIFICATION,
        has_header=_has_header,
    )
    print(colored("1. Finish loading dataset.", "red"))

    # 2. Run PSI
    print(colored("2. PSI protocol started, computing...", "red"))
    passive_psi = PassiveCM20PSI(messenger=_messenger, logger=_logger)
    common_ids = passive_psi.run(passive_dataset.ids)
    print(f"length of common ids: {len(common_ids)}")
    passive_dataset.filter(common_ids)
    print(passive_dataset.get_dataset().shape)
    passive_trainset, passive_testset = TorchDataset.train_test_split(
        passive_dataset, test_size=_test_size
    )
    print(passive_trainset.get_dataset().shape, passive_testset.get_dataset().shape)
    print(colored("3. Finish psi protocol", "red"))

    # 3. VFL training
    print(colored("4. Passive party started training...", "red"))
    bottom_model = MLP(
        _bottom_nodes,
        activate_input=False,
        activate_output=True,
        random_state=_random_state,
    )
    cut_layer = CutLayer(*_cut_nodes, random_state=_random_state).to(_device)
    _models = {"bottom": bottom_model, "cut": cut_layer}
    _optimizers = {
        name: torch.optim.SGD(
            model.parameters(), lr=_learning_rate, momentum=0.9, weight_decay=5e-4
        )
        for name, model in _models.items()
    }

    # 4. Initialize vertical NN protocol and start fed training
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
    print(colored("5. Passive party finish vfl_nn training.", "red"))

    # 5. Close messenger, finish training
    _messenger.close()
