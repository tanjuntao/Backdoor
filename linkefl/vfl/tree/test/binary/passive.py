import pandas as pd

from linkefl.common.const import Const
from linkefl.common.factory import (
    logger_factory,
    messenger_factory,
    messenger_factory_disconnection,
)
from linkefl.dataio import NumpyDataset
from linkefl.vfl.tree.passive import PassiveTreeParty
class Test:
    @staticmethod
    def get_dataset(dataset_name, passive_feat_frac, feat_perm_option):
        print("Loading dataset...")
        passive_trainset = NumpyDataset.buildin_dataset(
            role=Const.PASSIVE_NAME,
            dataset_name=dataset_name,
            root="../../../data",
            train=True,
            download=True,
            passive_feat_frac=passive_feat_frac,
            feat_perm_option=feat_perm_option,
        )
        passive_testset = NumpyDataset.buildin_dataset(
            role=Const.PASSIVE_NAME,
            dataset_name=dataset_name,
            root="../../../data",
            train=False,
            download=True,
            passive_feat_frac=passive_feat_frac,
            feat_perm_option=feat_perm_option,
        )
        passive_trainset = NumpyDataset.feature_split(passive_trainset, n_splits=1)[0]
        passive_testset = NumpyDataset.feature_split(passive_testset, n_splits=1)[0]
        print("Done")

        return passive_trainset, passive_testset
    @staticmethod
    def get_messengers(active_ip, active_port, passive_ip, passive_port, drop_protection):
        if not drop_protection:
            messenger = messenger_factory(
                messenger_type=Const.FAST_SOCKET,
                role=Const.PASSIVE_NAME,
                active_ip=active_ip,
                active_port=active_port,
                passive_ip=passive_ip,
                passive_port=passive_port,
            )
        else:
            messenger = messenger_factory_disconnection(
                messenger_type=Const.FAST_SOCKET_V1,
                role=Const.PASSIVE_NAME,
                model_type="Tree",
                active_ip=active_ip,
                active_port=active_port,
                passive_ip=passive_ip,
                passive_port=passive_port,
            )
        return messenger

if __name__ == "__main__":
    # 0. Set parameters
    #  binary: cancer, digits, epsilon, census, credit, default_credit, criteo
    #  regression: diabetes
    dataset_name = "digits"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    task = "binary"
    crypto_type = Const.PLAIN   # Const.PLAIN, Const.FAST_PAILLIER
    colsample_bytree = 1
    saving_model = True
    model_dir = "../../models/test_binary"
    model_name = "passive.model"

    active_ip = "localhost"
    active_port = 21001
    passive_ip = "localhost"
    passive_port = 20001
    drop_protection = False
    reconnect_ports = 30003

    passive_trainset, passive_testset = Test.get_dataset(dataset_name, passive_feat_frac, feat_perm_option)
    messenger = Test.get_messengers(active_ip, active_port, passive_ip, passive_port, drop_protection)

    # 4. Initialize active tree party and start training
    logger = logger_factory(role=Const.PASSIVE_NAME)
    passive_party = PassiveTreeParty(
        task=task,
        crypto_type=crypto_type,
        messenger=messenger,
        logger=logger,
        saving_model=True,
        model_dir=model_dir,
        model_name=model_name,
    )

    passive_party.train(passive_trainset, passive_testset)

    # passive_party.online_inference(passive_testset, messenger, logger,
    #                                model_dir=model_dir,
    #                                model_name=model_name,)

    feature_importance_info = pd.DataFrame(
        passive_party.feature_importances_(importance_type="cover")
    )
    print(feature_importance_info)

    # 4. Close messenger, finish training
    messenger.close()