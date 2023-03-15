from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import Compose, ParseLabel
from linkefl.pipeline import PipeLine
from linkefl.psi import ActiveCM20PSI
from linkefl.vfl.tree import ActiveTreeParty

if __name__ == "__main__":
    # 0. Set parameters

    # dataloader
    trainset_path = (
        "/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/census-active-train.csv"
    )
    testset_path = (
        "/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/census-active-test.csv"
    )

    # messengers
    active_ips = ["localhost", "localhost"]
    active_ports = [20000, 20001]
    passive_ips = ["localhost", "localhost"]
    passive_ports = [30000, 30001]

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

    # logger
    logger = logger_factory(role=Const.ACTIVE_NAME)

    # transformer

    # model
    n_trees = 1
    task = "binary"
    n_labels = 2
    crypto_type = Const.FAST_PAILLIER
    learning_rate = 0.3
    compress = False
    max_bin = 16
    max_depth = 4
    reg_lambda = 0.1
    min_split_samples = 3
    min_split_gain = 1e-7
    fix_point_precision = 53
    sampling_method = "uniform"
    subsample = 1
    top_rate = 0.5
    other_rate = 0.5
    colsample_bytree = 1
    n_processes = 6
    saving_model = True
    model_name = "active_tree_model.model"

    crypto_system = crypto_factory(
        crypto_type=crypto_type,
        key_size=1024,
        num_enc_zeros=10000,
        gen_from_set=False,
    )

    # 1. Load dataset
    active_trainset = NumpyDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path=trainset_path,
        dataset_type=Const.CLASSIFICATION,
    )
    active_testset = NumpyDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path=testset_path,
        dataset_type=Const.CLASSIFICATION,
        mappings=active_trainset.mappings,
    )

    # 2. Build pipeline
    psi = ActiveCM20PSI(messengers=messengers, logger=logger)
    transforms = Compose([ParseLabel()])
    model = ActiveTreeParty(
        n_trees=n_trees,
        task=task,
        n_labels=n_labels,
        crypto_type=crypto_type,
        crypto_system=crypto_system,
        messengers=messengers,
        logger=logger,
        training_mode="xgboost",
        learning_rate=learning_rate,
        compress=compress,
        max_bin=max_bin,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        min_split_samples=min_split_samples,
        min_split_gain=min_split_gain,
        fix_point_precision=fix_point_precision,
        sampling_method=sampling_method,
        subsample=subsample,
        top_rate=top_rate,
        other_rate=other_rate,
        colsample_bytree=colsample_bytree,
        n_processes=n_processes,
        saving_model=saving_model,
        model_name=model_name,
    )

    pipeline = PipeLine([psi, transforms, model], role=Const.ACTIVE_NAME)

    # 3. Trigger pipeline
    pipeline.fit(active_trainset, active_testset)
    pipeline.score(active_testset)
