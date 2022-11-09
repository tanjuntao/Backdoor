from linkefl.common.const import Const
from linkefl.common.factory import logger_factory, messenger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import Compose, ParseLabel
from linkefl.pipeline import PipeLine
from linkefl.psi import CM20PSIPassive
from linkefl.vfl.tree import PassiveTreeParty

if __name__ == "__main__":
    # 0. Set parameters

    # dataloader
    trainset_path = "census-passive2-train.csv"
    testset_path = "census-passive2-test.csv"

    # messenger
    active_ip = "localhost"
    active_port = 30000
    passive_ip = "localhost"
    passive_port = 30001

    messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    # logger
    logger = logger_factory(role=Const.PASSIVE_NAME)

    # psi
    log_height = 8
    width = 632
    hash_length = 10
    h1_length = 32
    bucket1 = 1 << 8
    bucket2 = 1 << 8

    # transformer

    # model
    task = "binary"
    crypto_type = Const.FAST_PAILLIER
    max_bin = 16
    colsample_bytree = 1

    # 1. Load dataset
    passive_trainset = NumpyDataset.from_csv(
        role=Const.PASSIVE_NAME,
        abs_path=trainset_path,
        dataset_type=Const.CLASSIFICATION,
    )
    passive_testset = NumpyDataset.from_csv(
        role=Const.PASSIVE_NAME,
        abs_path=testset_path,
        dataset_type=Const.CLASSIFICATION,
        mappings=passive_trainset.mappings,
    )

    # 2. Build pipeline
    psi = CM20PSIPassive(
        messenger,
        logger,
        log_height=log_height,
        width=width,
        hash_length=hash_length,
        h1_length=h1_length,
        bucket1=bucket1,
        bucket2=bucket2,
    )
    transforms = Compose([ParseLabel()])
    model = PassiveTreeParty(
        task=task,
        crypto_type=crypto_type,
        messenger=messenger,
        max_bin=max_bin,
        colsample_bytree=colsample_bytree,
    )

    pipeline = PipeLine([psi, transforms, model], role=Const.PASSIVE_NAME)

    # 3. Trigger pipeline
    pipeline.fit(passive_trainset, passive_testset)
    pipeline.score(passive_testset)
