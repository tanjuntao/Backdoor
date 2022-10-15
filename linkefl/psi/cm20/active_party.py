import random
import time
from typing import List

import numpy as np

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket

try:
    from linkefl.psi.cm20.PsiPython import PsiReceiver
except ImportError:
    raise ImportError("Please build CM20 and put it under linkefl.psi.cm20")


class CM20PSIActive:
    def __init__(
        self,
        ids: List,
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
        self.ids = ids
        self.messenger = messenger
        self.logger = logger
        self.log_height = log_height
        self.width = width
        self.hash_length = hash_length
        self.h1_length = h1_length
        self.bucket1 = bucket1
        self.bucket2 = bucket2

        self.active_ids_len = len(ids)
        self.passive_ids_len = None

    def run(self):
        start = time.time()

        # 1. sync seed and number of ids
        seed = random.randint(0, 1 << 32)
        self.logger.log(f"Common seed: {seed}")
        self.messenger.send(seed)
        self.messenger.send(self.active_ids_len)
        self.passive_ids_len = self.messenger.recv()

        # 2. compute common ids and return indexes
        common_indexes = PsiReceiver().run(
            self.messenger.passive_ip,
            seed,
            self.passive_ids_len,
            self.active_ids_len,
            1 << self.log_height,
            self.log_height,
            self.width,
            self.ids,
            self.hash_length,
            self.h1_length,
            self.bucket1,
            self.bucket1,
        )

        # 3. find the intersection
        intersections = np.array(self.ids)[np.array(common_indexes)]
        self.messenger.send(intersections)

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
    # 1. get sample IDs
    _ids = gen_dummy_ids(size=10_000_000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20000,
        passive_ip="127.0.0.1",
        passive_port=30000,
    )
    _logger = logger_factory(role=Const.ACTIVE_NAME)

    # 3. Start the CM20 protocol
    active_party = CM20PSIActive(_ids, _messenger, _logger)
    intersections_ = active_party.run()
    print(intersections_[:10])

    # 4. Close messenger
    _messenger.close()
