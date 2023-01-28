from linkefl.common.const import Const
from linkefl.common.factory import logger_factory, messenger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import Compose, ParseLabel
from linkefl.pipeline import PipeLine
from linkefl.psi import RSAPSIPassive
from linkefl.vfl.tree import PassiveTreeParty

if __name__ == "__main__":
    # 0. Set parameters

    # dataloader
    trainset_path = "census-passive1-train.csv"
    testset_path = "census-passive1-test.csv"

    # messenger
    active_ip = "localhost"
    active_port = 20000
    passive_ip = "localhost"
    passive_port = 20001

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

    # transformer

    # model
    task = "binary"
    crypto_type = Const.FAST_PAILLIER
    max_bin = 16
    colsample_bytree = 1
    saving_model = True
    model_name = "passive1_tree_model.model"

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
    psi = RSAPSIPassive(messenger, logger)
    transforms = Compose([ParseLabel()])
    model = PassiveTreeParty(
        task=task,
        crypto_type=crypto_type,
        messenger=messenger,
        logger=logger,
        max_bin=max_bin,
        colsample_bytree=colsample_bytree,
        saving_model=saving_model,
        model_name=model_name,
    )
    pipeline = PipeLine([psi, transforms, model], role=Const.PASSIVE_NAME)

    # 3. Trigger pipeline
    pipeline.fit(passive_trainset, passive_testset)
    pipeline.score(passive_testset)
