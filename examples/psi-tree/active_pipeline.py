from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import Compose, ParseLabel
from linkefl.messenger import FastSocket
from linkefl.pipeline import PipeLine
from linkefl.psi import CM20PSIActive
from linkefl.vfl.tree import ActiveTreeParty

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
        role=Const.ACTIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    # logger
    logger = logger_factory(role=Const.ACTIVE_NAME)

    # transformer

    # model
    n_trees = 1
    task = "binary"
    n_labels = 2
    crypto_type = Const.FAST_PAILLIER
    n_processes = 6

    crypto_system = crypto_factory(
        crypto_type=crypto_type,
        key_size=1024,
        num_enc_zeros=10000,
        gen_from_set=False,
    )

    # 1. Load dataset
    active_trainset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root="data",
        train=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
        download=True,
    )
    active_testset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root="data",
        train=False,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
        download=True,
    )

    # 2. Build pipeline
    psi = CM20PSIActive(messenger, logger)
    transforms = Compose([ParseLabel()])
    model = ActiveTreeParty(
        n_trees=n_trees,
        task=task,
        n_labels=n_labels,
        crypto_type=crypto_type,
        crypto_system=crypto_system,
        messengers=[messenger],
        n_processes=n_processes,
    )

    pipeline = PipeLine([psi, transforms, model], role=Const.ACTIVE_NAME)

    # 3. Trigger pipeline
    pipeline.fit(active_trainset, active_testset)
    pipeline.score(active_testset)
