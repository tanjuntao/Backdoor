from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import RSACrypto
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

    task = "binary"
    crypto_type = Const.FAST_PAILLIER

    messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )
    logger = logger_factory(role=Const.PASSIVE_NAME)
    n_processes = 6
    psi_crypto_system = RSACrypto()

    data_reader = NumpyDataset_from_csv_ReaderComponent(
        role=Const.PASSIVE_NAME,
        trainset_path=(
            r"../../linkefl/data/tabular/give_me_some_credit_passive_train.csv"
        ),
        testset_path=r"../../linkefl/data/tabular/give_me_some_credit_passive_test.csv",
        dataset_type=Const.CLASSIFICATION,
    )

    data_transform = TransformComponent(
        role=Const.PASSIVE_NAME,
        # trainset_transform=parse_label,
        # testset_transform=parse_label,
    )

    data_psi = RSAPSIComponent(
        role=Const.PASSIVE_NAME,
        messenger=messenger,
        logger=logger,
        n_processes=n_processes,
        crypto_system=psi_crypto_system,
    )

    model = VFLSBTComponent.passive(
        task=task,
        crypto_type=crypto_type,
        messenger=messenger,
    )

    pipeline = PipeLine()
    pipeline.add_component(data_reader)
    pipeline.add_component(data_transform)
    pipeline.add_component(data_psi)
    pipeline.add_component(model)
    pipeline.run()
    print()
