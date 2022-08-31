from linkefl.common.const import Const
from linkefl.common.factory import messenger_factory
from linkefl.dataio import NumpyDataset
from linkefl.vfl.tree import PassiveTreeParty


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "credit"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    task = "binary"
    _crypto_type = Const.PAILLIER

    active_ip = "localhost"
    active_port = 20001
    passive_ip = "localhost"
    passive_port = 30001

    # 1. Load datasets
    print("Loading dataset...")
    passive_trainset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=dataset_name,
        train=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_testset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=dataset_name,
        train=False,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_trainset, _ = NumpyDataset.feature_split(passive_trainset, 2)
    passive_testset, _ = NumpyDataset.feature_split(passive_testset, 2)
    print("Done")

    # 2. Initialize messenger
    messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    # 3. Initialize passive tree party and start training
    passive_party = PassiveTreeParty(task=task, crypto_type=_crypto_type, messenger=messenger)
    passive_party.train(passive_trainset, passive_testset)

    # 4. Close messenger, finish training
    messenger.close()
