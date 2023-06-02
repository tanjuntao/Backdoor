from linkefl.common.const import Const
from linkefl.common.factory import (
    crypto_factory,
    logger_factory,
    messenger_factory,
    messenger_factory_disconnection,
)
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import parse_label
from linkefl.vfl.tree.active import ActiveTreeParty

class Test:
    @staticmethod
    def get_dataset(dataset_name, passive_feat_frac, feat_perm_option):
        print("Loading dataset...")
        active_trainset = NumpyDataset.buildin_dataset(
            role=Const.ACTIVE_NAME,
            dataset_name=dataset_name,
            root="../../../data",
            train=True,
            download=True,
            passive_feat_frac=passive_feat_frac,
            feat_perm_option=feat_perm_option,
        )
        active_testset = NumpyDataset.buildin_dataset(
            role=Const.ACTIVE_NAME,
            dataset_name=dataset_name,
            root="../../../data",
            train=False,
            download=True,
            passive_feat_frac=passive_feat_frac,
            feat_perm_option=feat_perm_option,
        )
        active_trainset = parse_label(active_trainset)
        active_testset = parse_label(active_testset)
        print("Done")

        return active_trainset, active_testset

    @staticmethod
    def get_crypto_system(crypto_type, key_size):
        crypto_system = crypto_factory(
            crypto_type=crypto_type,
            key_size=key_size,
            num_enc_zeros=2000,
            gen_from_set=False,
        )
        return crypto_system
    @staticmethod
    def get_messengers(active_ips, active_ports, passive_ips, passive_ports, drop_protection):
        if not drop_protection:
            messengers = [
                messenger_factory(
                    messenger_type=Const.FAST_SOCKET,
                    role=Const.ACTIVE_NAME,
                    active_ip=active_ip,
                    active_port=active_port,
                    passive_ip=passive_ip,
                    passive_port=passive_port,
                )
                for active_ip, active_port, passive_ip, passive_port in zip(
                    active_ips, active_ports, passive_ips, passive_ports
                )
            ]
        else:
            messengers = [
                messenger_factory_disconnection(
                    messenger_type=Const.FAST_SOCKET_V1,
                    role=Const.ACTIVE_NAME,
                    model_type="Tree",  # used as tag to verify data
                    active_ip=active_ip,
                    active_port=active_port,
                    passive_ip=passive_ip,
                    passive_port=passive_port,
                )
                for active_ip, active_port, passive_ip, passive_port in zip(
                    active_ips, active_ports, passive_ips, passive_ports
                )
            ]
        return messengers

if __name__ == "__main__":
    # 0. Set parameters
    #  binary: cancer, digits, epsilon, census, credit, default_credit, criteo
    #  regression: diabetes
    dataset_name = "digits"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    n_trees = 5
    task = "binary"
    n_labels = 2
    key_size = 1024

    crypto_type = Const.FAST_PAILLIER           # Const.PLAIN, Const.FAST_PAILLIER
    compress = True                    # False, True
    training_mode = "lightgbm"          # "lightgbm", "xgboost"
    sampling_method = "uniform"         # "uniform", "goss"
    max_depth = 5
    max_num_leaves = 10
    subsample = 1
    top_rate = 0.3
    other_rate = 0.7
    colsample_bytree = 0.8
    saving_model = True
    model_dir = "../../models/test_binary"
    model_name = "active.model"


    active_ips = ["localhost"]
    active_ports = [21001]
    passive_ips = ["localhost"]
    passive_ports = [20001]
    drop_protection = False
    reconnect_ports = [30003]

    active_trainset, active_testset = Test.get_dataset(dataset_name, passive_feat_frac, feat_perm_option)
    crypto_system = Test.get_crypto_system(crypto_type, key_size)
    messengers = Test.get_messengers(active_ips, active_ports, passive_ips, passive_ports, drop_protection)

    # 4. Initialize active tree party and start training
    logger = logger_factory(role=Const.ACTIVE_NAME)
    active_party = ActiveTreeParty(
        n_trees=n_trees,
        task=task,
        n_labels=n_labels,
        crypto_type=crypto_type,
        crypto_system=crypto_system,
        messengers=messengers,
        logger=logger,
        compress=compress,
        training_mode=training_mode,
        sampling_method=sampling_method,
        max_depth=max_depth,
        max_num_leaves=max_num_leaves,
        subsample=subsample,
        top_rate=top_rate,
        other_rate=other_rate,
        colsample_bytree=colsample_bytree,
        n_processes=6,
        drop_protection=drop_protection,
        reconnect_ports=reconnect_ports,
        saving_model=True,
        model_dir=model_dir,
        model_name=model_name,
    )

    active_party.train(active_trainset, active_testset)

    # preds = ActiveTreeParty.online_inference(
    #     active_testset,
    #     messengers,
    #     logger,
    #     model_dir=model_dir,
    #     model_name=model_name,
    # )
    # print(preds)

    # feature_importance_info = pd.DataFrame(
    #     active_party.feature_importances_(importance_type='cover')
    # )
    # print(feature_importance_info)

    # active_party.print_model_structure(tree_structure="VERTICAL")

    # 5. Close messengers, finish training
    for messenger in messengers:
        messenger.close()
