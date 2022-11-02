import pandas as pd

from linkefl.common.const import Const
from linkefl.common.factory import messenger_factory, messenger_factory_disconnection
from linkefl.dataio import NumpyDataset
from linkefl.vfl.tree import PassiveTreeParty


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "cancer"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    task = "binary"
    _crypto_type = Const.FAST_PAILLIER

    active_ip = "localhost"
    active_port = 20001
    passive_ip = "localhost"
    passive_port = 30001

    # 1. Load datasets
    print("Loading dataset...")
    passive_trainset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
                                                    dataset_name=dataset_name,
                                                    root='../data',
                                                    train=True,
                                                    download=True,
                                                    passive_feat_frac=passive_feat_frac,
                                                    feat_perm_option=feat_perm_option)
    passive_testset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
                                                   dataset_name=dataset_name,
                                                   root='../data',
                                                   train=False,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    passive_trainset, _ = NumpyDataset.feature_split(passive_trainset, 2)
    passive_testset, _ = NumpyDataset.feature_split(passive_testset, 2)
    # passive_trainset = NumpyDataset.feature_split(passive_trainset, 1)
    # passive_testset = NumpyDataset.feature_split(passive_testset, 1)
    print("Done")

    # 2. Initialize messenger
    # messenger = messenger_factory(
    #     messenger_type=Const.FAST_SOCKET,
    #     role=Const.PASSIVE_NAME,
    #     active_ip=active_ip,
    #     active_port=active_port,
    #     passive_ip=passive_ip,
    #     passive_port=passive_port,
    # )
    messenger = messenger_factory_disconnection(
        messenger_type=Const.FAST_SOCKET_V1,
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    # 3. Initialize passive tree party and start training
    passive_party = PassiveTreeParty(
        client_name="1",
        task=task,
        crypto_type=_crypto_type,
        messenger=messenger,
        saving_model=True,
    )
    passive_party.train(passive_trainset, passive_testset)
    # passive_party.online_inference(passive_testset, "xxx.model")

    # test
    feature_importance_info = pd.DataFrame(passive_party.feature_importances_(importance_type='cover'))
    print(feature_importance_info)

    # 4. Close messenger, finish training
    messenger.close()
