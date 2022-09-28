from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory
from linkefl.crypto import RSACrypto
from linkefl.feature import parse_label
from linkefl.messenger import FastSocket
from linkefl.pipeline import PipeLine
from linkefl.pipeline.component import (
    NumpyDataset_from_csv_ReaderComponent,
    RSAPSIComponent,
    TransformComponent,
    VFLSBTComponent,
)

if __name__ == "__main__":
    active_ip = "localhost"
    active_port = 20000
    passive_ip = "localhost"
    passive_port = 30000

    n_trees = 5
    task = "binary"
    n_labels = 2
    crypto_type = Const.FAST_PAILLIER
    key_size = 1024
    model_crypto_system = crypto_factory(
        crypto_type=crypto_type,
        key_size=key_size,
        num_enc_zeros=10000,
        gen_from_set=False,
    )

    messenger = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )
    logger = logger_factory(role=Const.ACTIVE_NAME)
    n_processes = 6
    psi_crypto_system = RSACrypto()

    data_reader = NumpyDataset_from_csv_ReaderComponent(
        role=Const.ACTIVE_NAME,
        trainset_path=(
            r"../../linkefl/data/tabular/give_me_some_credit_active_train.csv"
        ),
        testset_path=r"../../linkefl/data/tabular/give_me_some_credit_active_test.csv",
        dataset_type=Const.CLASSIFICATION,
    )

    data_transform = TransformComponent(
        role=Const.ACTIVE_NAME,
        trainset_transform=parse_label,
        testset_transform=parse_label,
    )

    data_psi = RSAPSIComponent(
        role=Const.ACTIVE_NAME,
        messenger=messenger,
        logger=logger,
        n_processes=n_processes,
        crypto_system=psi_crypto_system,
    )

    model = VFLSBTComponent.active(
        n_trees=n_trees,
        task=task,
        n_labels=n_labels,
        crypto_type=crypto_type,
        crypto_system=model_crypto_system,
        messenger=messenger,
    )

    pipeline = PipeLine()
    pipeline.add_component(data_reader)
    pipeline.add_component(data_transform)
    pipeline.add_component(data_psi)
    pipeline.add_component(model)
    pipeline.run()
    print()
