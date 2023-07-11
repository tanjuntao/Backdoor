import torch
from termcolor import colored
from torch import nn

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory, messenger_factory
from linkefl.dataio import TorchDataset
from linkefl.modelzoo import MLP, CutLayer
from linkefl.psi import ActiveCM20PSI
from linkefl.vfl.nn import ActiveNeuralNetwork

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_path = "./data/tabmnist-active000.csv"
    _has_header = False
    _test_size = 0.2
    _active_ips = ["localhost"]
    _active_ports = [20000]
    _passive_ips = ["localhost"]
    _passive_ports = [30000]
    _epochs = 50
    _batch_size = 256
    _learning_rate = 0.01
    _loss_fn = nn.CrossEntropyLoss()
    _random_state = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _saving_model = True
    _bottom_nodes = [392, 256, 128]
    _cut_nodes = [128, 64]
    _top_nodes = [64, 10]
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _messengers = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=ac_ip,
            active_port=ac_port,
            passive_ip=pass_ip,
            passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in zip(
            _active_ips, _active_ports, _passive_ips, _passive_ports
        )
    ]

    # 1. Load dataset
    active_dataset = TorchDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path=_dataset_path,
        dataset_type=Const.CLASSIFICATION,
        has_header=_has_header,
    )
    print(colored("1. Finish loading dataset.", "red"))

    # 2. Run PSI
    print(colored("2. PSI protocol started, computing...", "red"))
    active_psi = ActiveCM20PSI(messengers=_messengers, logger=_logger)
    common_ids = active_psi.run(active_dataset.ids)
    print(f"lenght of common ids: {len(common_ids)}")
    active_dataset.filter(common_ids)
    print(active_dataset.get_dataset().shape)
    active_trainset, active_testset = TorchDataset.train_test_split(
        active_dataset, test_size=_test_size
    )
    print(active_trainset.get_dataset().shape, active_testset.get_dataset().shape)
    print(colored("3. Finish psi protocol", "red"))

    # 3. VFL training
    print(colored("4. Active party started training...", "red"))
    bottom_model = MLP(
        _bottom_nodes,
        activate_input=False,
        activate_output=True,
        random_state=_random_state,
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

    # 4. Initialize vertical NN protocol and start training
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
    )
    active_party.train(active_trainset, active_testset)
    print(colored("5. Active party finished vfl_nn training.", "red"))

    # 5. Close messenger, finish training
    for messenger in _messengers:
        messenger.close()
