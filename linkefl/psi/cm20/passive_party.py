import time
from typing import Union

from linkefl.base import BasePSIComponent
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset, TorchDataset

try:
    from linkefl.psi.cm20.PsiPython import PsiSender
except ImportError:
    raise RuntimeError(
        "Script launching order error. You should launch active party first."
    )


class CM20PSIPassive(BasePSIComponent):
    def __init__(
        self,
        messenger,
        logger,
        *,
        log_height=8,
        width=632,
        hash_length=10,
        h1_length=32,
        bucket1=1 << 8,
        bucket2=1 << 8,
    ):
        self.messenger = messenger
        self.logger = logger
        self.log_height = log_height
        self.width = width
        self.hash_length = hash_length
        self.h1_length = h1_length
        self.bucket1 = bucket1
        self.bucket2 = bucket2

    def fit(self, dataset: Union[NumpyDataset, TorchDataset], role=Const.PASSIVE_NAME):
        ids = dataset.ids
        intersections = self.run(ids)
        dataset.filter(intersections)

        return dataset

    def run(self, ids):
        passive_ids_len = len(ids)

        start = time.time()
        self.messenger.send(Const.START_SIGNAL)  # send starting signal

        # 1. sync seed and number of ids
        seed = self.messenger.recv()
        active_ids_len = self.messenger.recv()
        self.messenger.send(passive_ids_len)

        # 2. compute common ids
        PsiSender().run(
            self.messenger.passive_ip,
            seed,
            passive_ids_len,
            active_ids_len,
            1 << self.log_height,
            self.log_height,
            self.width,
            ids,
            self.hash_length,
            self.h1_length,
            self.bucket1,
            self.bucket1,
        )

        # 3. receive intersection
        intersections = self.messenger.recv()

        self.logger.log("Size of intersection: {}".format(len(intersections)))

        self.logger.log(
            "Total protocol execution time: {:.5f}".format(time.time() - start)
        )
        self.logger.log_component(
            name=Const.CM20_PSI,
            status=Const.SUCCESS,
            begin=start,
            end=time.time(),
            duration=time.time() - start,
            progress=1.0,
        )
        return intersections


if __name__ == "__main__":
    from linkefl.common.factory import logger_factory
    from linkefl.dataio import gen_dummy_ids
    from linkefl.messenger import FastSocket

    # 1. Get sample IDs
    _ids = gen_dummy_ids(size=10_000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20000,
        passive_ip="127.0.0.1",
        passive_port=30000,
    )
    _logger = logger_factory(role=Const.PASSIVE_NAME)

    # 3. Start the CM20 protocol
    passive_party = CM20PSIPassive(_messenger, _logger)
    intersections_ = passive_party.run(_ids)
    print(len(intersections_))

    # 4. Close messenger
    _messenger.close()
