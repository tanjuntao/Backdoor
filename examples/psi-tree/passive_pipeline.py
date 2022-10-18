from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature import Compose, ParseLabel
from linkefl.messenger import FastSocket
from linkefl.pipeline import PipeLine
from linkefl.psi import CM20PSIPassive
from linkefl.vfl.tree import PassiveTreeParty

if __name__ == "__main__":
    # 0. Set parameters

    # dataloader
    dataset_name = "credit"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    # messenger
    active_ip = "localhost"
    active_port = 20000
    passive_ip = "localhost"
    passive_port = 30000

    messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    # logger
    logger = logger_factory(role=Const.PASSIVE_NAME)

    # transformer

    # model
    task = "binary"
    crypto_type = Const.FAST_PAILLIER

    # 1. Load dataset
    passive_trainset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=dataset_name,
        root="data",
        train=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
        download=True,
    )
    passive_testset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=dataset_name,
        root="data",
        train=False,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
        download=True,
    )

    # 2. Build pipeline
    psi = CM20PSIPassive(messenger, logger)
    transforms = Compose([ParseLabel()])
    model = PassiveTreeParty(
        task=task,
        crypto_type=crypto_type,
        messenger=messenger,
    )

    pipeline = PipeLine([psi, transforms, model], role=Const.PASSIVE_NAME)

    # 3. Trigger pipeline
    pipeline.fit(passive_trainset, passive_testset)
    pipeline.score(passive_testset)
