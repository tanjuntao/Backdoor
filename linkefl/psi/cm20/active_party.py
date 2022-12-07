import os
import random
import sys
import time
from typing import Union
from urllib.error import URLError

import numpy as np

from linkefl.base import BasePSIComponent
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset, TorchDataset

try:
    from linkefl.psi.cm20.PsiPython import PsiReceiver
except ImportError:
    from linkefl.util import urlretrive

    resources = {
        "37-darwin": "PsiPython.cpython-37m-darwin.so",
        "37-linux": "PsiPython.cpython-37m-x86_64-linux-gnu.so",
        "38-darwin": "PsiPython.cpython-38-darwin.so",
        "38-linux": "PsiPython.cpython-38-x86_64-linux-gnu.so",
        "39-darwin": "PsiPython.cpython-39-darwin.so",
        "39-linux": "PsiPython.cpython-39-x86_64-linux-gnu.so",
        "310-darwin": "PsiPython.cpython-310-darwin.so",
        "310-linux": "PsiPython.cpython-310-x86_64-linux-gnu.so",
    }
    py_version = str(sys.version_info[0]) + str(sys.version_info[1])
    platform = sys.platform
    if platform not in ('linux', 'darwin'):
        raise RuntimeError('Currently only Linux and macOS are supported OS platform, '
                           'but you are in a {} platform.'.format(platform))
    key = py_version + '-' + platform
    filename = resources[key]
    this_directory = os.path.abspath(os.path.dirname(__file__))
    fpath = os.path.join(this_directory, filename)
    base_url = "http://47.96.163.59:80/wheels/"
    full_url = base_url + filename
    try:
        print('Downloading {} to {}'.format(full_url, fpath))
        urlretrive(full_url, fpath)
    except URLError as error:
        raise RuntimeError('Failed to download {} with error message: {}'
                           .format(full_url, error))
    # This static error message can be safely ignored if there is no downloading error.
    # If the downloading process completes successfully, there will always be a .so file
    # under the current directory that can be imported into this module.
    from linkefl.psi.cm20.PsiPython import PsiReceiver
    print('Done!')


class CM20PSIActive(BasePSIComponent):
    def __init__(
        self,
        messengers,
        logger,
        *,
        log_height=8,
        width=632,
        hash_length=10,
        h1_length=32,
        bucket1=1 << 8,
        bucket2=1 << 8,
    ):
        self.messengers = messengers
        self.logger = logger
        self.log_height = log_height
        self.width = width
        self.hash_length = hash_length
        self.h1_length = h1_length
        self.bucket1 = bucket1
        self.bucket2 = bucket2

    def fit(self, dataset: Union[NumpyDataset, TorchDataset], role=Const.ACTIVE_NAME):
        ids = dataset.ids
        intersections = self.run(ids)
        dataset.filter(intersections)

        return dataset

    def run(self, ids):
        start = time.time()

        for messenger in self.messengers:
            active_ids_len = len(ids)

            # self.logger.log('Active party starts CM20 PSI, listening...')
            signal = messenger.recv()
            if signal != Const.START_SIGNAL:
                raise RuntimeError("Invalid start signal from passive party.")

            # 1. sync seed and number of ids
            seed = random.randint(0, 1 << 32)
            self.logger.log(f"Common seed: {seed}")
            messenger.send(seed)
            messenger.send(active_ids_len)
            passive_ids_len = messenger.recv()

            # 2. compute common ids and return indexes
            time.sleep(2)
            common_indexes = PsiReceiver().run(
                messenger.passive_ip,
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

            ids = np.array(ids)[np.array(common_indexes)].tolist()

        # 3. find the intersection
        intersections = ids
        for messenger in self.messengers:
            messenger.send(intersections)

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
    from linkefl.dataio import gen_dummy_ids, NumpyDataset, TorchDataset
    from linkefl.messenger import FastSocket

    # 1. get sample IDs
    _ids = gen_dummy_ids(size=50_000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger1 = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20000,
        passive_ip="127.0.0.1",
        passive_port=30000,
    )
    _messenger2 = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20001,
        passive_ip="127.0.0.1",
        passive_port=30001,
    )
    _logger = logger_factory(role=Const.ACTIVE_NAME)

    # 3. Start the CM20 protocol
    active_party = CM20PSIActive([_messenger1, _messenger2], _logger)
    intersections_ = active_party.run(_ids)
    print(len(intersections_))

    # 4. Close messenger
    _messenger1.close()
    _messenger2.close()
