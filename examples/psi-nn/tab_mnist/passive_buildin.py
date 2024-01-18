import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
from linkefl.dataio import TorchDataset
from linkefl.modelzoo.mlp import MLP, CutLayer
from linkefl.util import num_input_nodes
from linkefl.vfl.nn import PassiveNeuralNetwork

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_name = "tab_mnist"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _active_ip = "localhost"
    _active_port = 20000
    _passive_ip = "localhost"
    _passive_port = 30000
    _epochs = 50
    _batch_size = 128
    _learning_rate = 0.1
    _crypto_type = Const.PLAIN
    _key_size = 1024
    _num_workers = 1
    _random_state = None
    _device = "cpu"
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    _messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
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

    # 1. Load datasets
    print("Loading dataset...")
    passive_trainset = TorchDataset.buildin_dataset(
        dataset_name=_dataset_name,
        role=Const.PASSIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
    )
    passive_testset = TorchDataset.buildin_dataset(
        dataset_name=_dataset_name,
        role=Const.PASSIVE_NAME,
        root="../data",
        train=False,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
        seed=_random_state,
    )
    print("Done.")

    # 2. Create PyTorch models and optimizers
    input_nodes = num_input_nodes(
        dataset_name=_dataset_name,
        role=Const.PASSIVE_NAME,
        passive_feat_frac=_passive_feat_frac,
    )
    # # mnist & fashion_mnist
    bottom_nodes = [input_nodes, 256, 128, 128]
    cut_nodes = [128, 64]

    _bottom_model = MLP(
        bottom_nodes,
        activate_input=False,
        activate_output=True,
        random_state=_random_state,
    ).to(_device)
    _cut_layer = CutLayer(*cut_nodes, random_state=_random_state).to(_device)
    _models = {"bottom": _bottom_model, "cut": _cut_layer}
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
        num_workers=_num_workers,
        device=_device,
        random_state=_random_state,
        saving_model=True,
        schedulers=schedulers,
        model_dir="../models/tab_mnist",
        model_name="VFL_passive.model",
    )
    passive_party.train(passive_trainset, passive_testset)

    # 4. Close messenger, finish training
    _messenger.close()
