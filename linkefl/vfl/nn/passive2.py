import torch

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import MediaDataset, TorchDataset  # noqa: F403
from linkefl.vfl.nn.passive import PassiveNeuralNetwork

if __name__ == "__main__":
    from linkefl.common.factory import crypto_factory, messenger_factory
    from linkefl.modelzoo.mlp import MLP, CutLayer
    from linkefl.util import num_input_nodes

    # 0. Set parameters
    _dataset_name = "tab_mnist"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _active_ip = "localhost"
    _active_port = 20001
    _passive_ip = "localhost"
    _passive_port = 30001
    _epochs = 10
    _batch_size = 100
    _learning_rate = 0.001
    _crypto_type = Const.PLAIN
    _key_size = 1024
    _num_workers = 1
    _random_state = None
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _saving_model = True
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
    passive_trainset = TorchDataset.feature_split(passive_trainset, n_splits=2)[1]
    passive_testset = TorchDataset.feature_split(passive_testset, n_splits=2)[1]

    print("Done.")

    # 2. Create PyTorch models and optimizers
    input_nodes = num_input_nodes(
        dataset_name=_dataset_name,
        role=Const.PASSIVE_NAME,
        passive_feat_frac=_passive_feat_frac,
    )
    # # mnist & fashion_mnist
    bottom_nodes = [int(input_nodes / 2), 256, 128]
    cut_nodes = [128, 64]

    _bottom_model = MLP(
        bottom_nodes,
        activate_input=False,
        activate_output=True,
        random_state=_random_state,
    ).to(_device)
    # bottom_model = ResNet18(in_channel=1).to(_device)
    _cut_layer = CutLayer(*cut_nodes, random_state=_random_state).to(_device)
    _models = {"bottom": _bottom_model, "cut": _cut_layer}
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
        num_workers=_num_workers,
        device=_device,
        random_state=_random_state,
        saving_model=_saving_model,
    )
    passive_party.train(passive_trainset, passive_testset)

    # 4. Close messenger, finish training
    _messenger.close()
