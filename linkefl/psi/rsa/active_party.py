import hashlib
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Union

from termcolor import colored

from linkefl.base import BaseMessenger, BasePSIComponent
from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.common.log import GlobalLogger
from linkefl.crypto import RSA
from linkefl.dataio import NumpyDataset, TorchDataset


class RSAPSIActive(BasePSIComponent):
    def __init__(
        self,
        messenger: List[BaseMessenger],
        cryptosystem: RSA,
        logger: GlobalLogger,
        num_workers: int = -1,
        obfuscated_rate: float = 0,
    ):
        self.messenger = messenger
        self.cryptosystem = cryptosystem
        self.logger = logger
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.obfuscated_rate = obfuscated_rate

        self.HASHED_IDS_FILENAME = "hashed_signed_ids.pkl"
        self.HERE = os.path.abspath(os.path.dirname(__file__))

    def fit(self, dataset: Union[NumpyDataset, TorchDataset], role=Const.ACTIVE_NAME):
        ids = dataset.ids
        intersections = self.run(ids)
        dataset.filter(intersections)

        return dataset

    def run(self, ids):
        # 1. sync RSA public key
        self.logger.log("Active party starts PSI, listening...")
        self._sync_pubkey()
        start = time.time()

        # 2. signing blinded ids of passive party
        for msger in self.messenger:
            blinded_ids = msger.recv()
            signed_blinded_ids = self.cryptosystem.sign_vector(
                blinded_ids, using_pool=True, n_workers=self.num_workers
            )
            msger.send(signed_blinded_ids)
        self.logger.log("Active party sends signed blinded ids back to passive party.")
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.RUNNING,
            begin=start,
            end=None,
            duration=time.time() - start,
            progress=0.4,
        )

        # 3. signing and hashing its own ids
        signed_ids = self.cryptosystem.sign_vector(
            ids, using_pool=True, n_workers=self.num_workers
        )
        active_hashed_signed_ids = RSAPSIActive._hash_set(signed_ids)
        self.logger.log("Active party finished signing and hashing its own ids.")
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.RUNNING,
            begin=start,
            end=None,
            duration=time.time() - start,
            progress=0.8,
        )

        # 4. receiving hashed signed ids from passive party
        if len(self.messenger) == 1:  # single passive party
            passive_hashed_signed_ids = self.messenger[0].recv()
        else:  # multi passive parties (>=2)
            passive_hashed_signed_ids = None
            for msger in self.messenger:
                curr_hashed_signed_ids = msger.recv()  # a python list
                curr_hashed_signed_ids = set(
                    curr_hashed_signed_ids
                )  # convert to python set
                if passive_hashed_signed_ids is None:
                    passive_hashed_signed_ids = curr_hashed_signed_ids
                    continue
                # compute python set intersection via & operator
                passive_hashed_signed_ids = (
                    passive_hashed_signed_ids & curr_hashed_signed_ids
                )
            passive_hashed_signed_ids = list(
                passive_hashed_signed_ids
            )  # convert back to list

        # 5. find the intersection
        intersections, intersection_hashed_ids = RSAPSIActive._intersect(
            ids, active_hashed_signed_ids, passive_hashed_signed_ids, self.obfuscated_rate
        )
        self.logger.log("Size of intersection: {}".format(len(intersections)))
        for msger in self.messenger:
            msger.send(intersection_hashed_ids)

        self.logger.log(
            "Total protocol execution time: {:.5f}".format(time.time() - start)
        )
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.SUCCESS,
            begin=start,
            end=time.time(),
            duration=time.time() - start,
            progress=1.0,
        )
        return intersections

    def run_offline(self, ids):
        print("[ACTIVE] Start the offline protocol...")
        begin = time.time()
        signed_ids = self.cryptosystem.sign_vector(
            ids, using_pool=True, n_workers=self.num_workers
        )
        print("Signing self id set time: {:.5f}".format(time.time() - begin))
        hashed_signed_ids = RSAPSIActive._hash_set(signed_ids)
        del signed_ids  # save momory

        target_dir = os.path.join(Path.home(), ".linkefl")
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        full_path = os.path.join(target_dir, self.HASHED_IDS_FILENAME)
        with open(full_path, "wb") as f:
            pickle.dump(hashed_signed_ids, f)
        print("[ACTIVE] Finish the offline protocol.")

    def run_online(self, ids):
        print("[ACTIVE] Started the online protocol, listening...")
        self._sync_pubkey()

        for msger in self.messenger:
            blinded_ids = msger.recv()
            begin = time.time()
            signed_blinded_ids = self.cryptosystem.sign_vector(
                blinded_ids, using_pool=True, n_workers=self.num_workers
            )
            print("Signing passive id set time: {:.5f}".format(time.time() - begin))
            msger.send(signed_blinded_ids)

        if len(self.messenger) == 1:  # single passive party
            passive_hashed_signed_ids = self.messenger[0].recv()
        else:  # multi passive parties (>=2)
            passive_hashed_signed_ids = None
            for msger in self.messenger:
                curr_hashed_signed_ids = msger.recv()  # a python list
                curr_hashed_signed_ids = set(
                    curr_hashed_signed_ids
                )  # convert to python set
                if passive_hashed_signed_ids is None:
                    passive_hashed_signed_ids = curr_hashed_signed_ids
                    continue
                # compute python set intersection via & operator
                passive_hashed_signed_ids = (
                    passive_hashed_signed_ids & curr_hashed_signed_ids
                )
            passive_hashed_signed_ids = list(
                passive_hashed_signed_ids
            )  # convert back to list
        full_path = os.path.join(Path.home(), ".linkefl", self.HASHED_IDS_FILENAME)
        with open(full_path, "rb") as f:
            active_hashed_signed_ids = pickle.load(f)
        begin = time.time()
        intersections, intersection_hashed_ids = RSAPSIActive._intersect(
            ids, active_hashed_signed_ids, passive_hashed_signed_ids, self.obfuscated_rate
        )
        del active_hashed_signed_ids  # save memory
        del passive_hashed_signed_ids  # save memory
        print("Intersection time: {}".format(time.time() - begin))
        print(colored("Size of intersection: {}".format(len(intersections)), "red"))
        for msger in self.messenger:
            msger.send(intersection_hashed_ids)

        os.remove(full_path)
        print("[ACTIVE] Finish the online protocol.")

        return intersections

    def _sync_pubkey(self):
        for msger in self.messenger:
            signal = msger.recv()
            if signal == Const.START_SIGNAL:
                n = self.cryptosystem.pub_key.n
                e = self.cryptosystem.pub_key.e
                msger.send([n, e])
            else:
                raise ValueError("Invalid start signal.")
        self.logger.log("[ACTIVE] Finish sending RSA public key.")

    @staticmethod
    def _hash_set(signed_set):
        return [hashlib.sha256(str(item).encode()).hexdigest() for item in signed_set]

    @staticmethod
    def _intersect(ids, active_hashed_ids, passive_hashed_ids, obfuscated_rate):
        intersections = []
        intersection_hashed_ids = []
        # directly convert Python list to Python set via the inbuild set() function is
        # faster than initilizing an empty set and then adding items sequentially
        # to this set.
        passive_hashed_set = set(passive_hashed_ids)
        for idx, hash_val in enumerate(active_hashed_ids):
            if hash_val in passive_hashed_set:
                intersections.append(ids[idx])
                intersection_hashed_ids.append(hash_val)

        if obfuscated_rate > 0:
            intersection_hashed_ids += random.choices(
                passive_hashed_ids,
                k=min(len(passive_hashed_ids), int(obfuscated_rate * len(intersections))),
            )
            intersection_hashed_ids = list(set(intersection_hashed_ids))

        # Before this function returns, the Python GC will delete the passive_hashed_set
        # which is REALLY time-consuming if the set size is big, e.g, >= 40 million.
        return intersections, intersection_hashed_ids


if __name__ == "__main__":
    import argparse

    from linkefl.dataio import NumpyDataset, TorchDataset, gen_dummy_ids
    from linkefl.messenger import FastSocket

    #   Option 1: split the protocol
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str)
    args = parser.parse_args()

    # # 1. get sample IDs
    # _ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)
    #
    # # 2. Initialize messenger
    # _messenger = FastSocket(role=Const.ACTIVE_NAME,
    #                         active_ip='127.0.0.1',
    #                         active_port=20000,
    #                         passive_ip='127.0.0.1',
    #                         passive_port=30000)
    # _logger = logger_factory(role=Const.ACTIVE_NAME)
    #
    # # 3. Start the RSA-Blind-Signature protocol
    # if args.phase == 'offline':
    #     _crypto = RSA()
    #     bob = RSAPSIActive([_messenger], _crypto, _logger)
    #     bob.run_offline(_ids)
    #
    # elif args.phase == 'online':
    #     _crypto = RSA.from_private_key()
    #     bob = RSAPSIActive([_messenger], _crypto, _logger)
    #     bob.run_online(_ids)
    #
    # else:
    #     raise ValueError(f"command line argument `--phase` can only"
    #                      f"take `offline` and `online`, "
    #                      f"but {args.phase} got instead")
    #
    # # 4. close messenger
    # _messenger.close()

    # '''
    #  Option 2: run the whole protocol
    # 1. get sample IDs
    _ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger1 = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20000,
        passive_ip="127.0.0.1",
        passive_port=30000,
    )
    # _messenger2 = FastSocket(role=Const.ACTIVE_NAME,
    #                          active_ip='127.0.0.1',
    #                          active_port=20001,
    #                          passive_ip='127.0.0.1',
    #                          passive_port=30001)
    # _messenger = [_messenger1, _messenger2]
    _messenger = [_messenger1]
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    # 3. Initialize cryptosystem
    _crypto = RSA()

    # 4. Start the RSA-Blind-Signature protocol
    active_party = RSAPSIActive(_messenger, _crypto, _logger, obfuscated_rate=0.5)
    intersections_ = active_party.run(_ids)
    print(len(intersections_))

    # 5. Close messenger
    for msger_ in _messenger:
        msger_.close()
    # '''
