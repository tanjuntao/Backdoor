import functools
import hashlib
import multiprocessing
import os
import pickle
import time
from pathlib import Path
from secrets import randbelow
from typing import List, Union

import gmpy2
from Crypto.PublicKey import RSA
from termcolor import colored

from linkefl.base import BaseMessenger, BasePSIComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.crypto import PartialRSA
from linkefl.dataio import NumpyDataset, TorchDataset


def _target_mp_pool(r, e, n):
    r_inv = gmpy2.invert(r, n)
    r_encrypted = gmpy2.powmod(r, e, n)

    return r_inv, r_encrypted


class PassiveRSAPSI(BasePSIComponent):
    def __init__(
        self,
        *,
        messenger: BaseMessenger,
        logger: GlobalLogger,
        num_workers: int = 1,
    ):
        self.messenger = messenger
        self.logger = logger
        assert (
            num_workers >= 1
        ), f"num_workers should >=1, but got {num_workers} instead."
        self.num_workers = num_workers

        self.RANDOMS_FILENAME = "randoms.pkl"
        self.LARGEST_RANDOM = pow(2, 512)
        self.HERE = os.path.abspath(os.path.dirname(__file__))

    def fit(
        self, dataset: Union[NumpyDataset, TorchDataset], role: str = Const.PASSIVE_NAME
    ) -> Union[NumpyDataset, TorchDataset]:
        ids = dataset.ids
        intersections = self.run(ids)
        dataset.filter(intersections)

        return dataset

    def run(self, ids: List[int]) -> List[int]:
        # sync RSA public key
        public_key = self._sync_pubkey()
        self.cryptosystem = PartialRSA(raw_public_key=public_key)
        start = time.time()

        # 1. generate random factors
        randoms = [randbelow(self.LARGEST_RANDOM) for _ in range(len(ids))]
        random_factors = self._random_factors_mp_pool(
            randoms, num_workers=self.num_workers
        )
        self.logger.log("Passive party finished genrating random factors.")

        # 2. blind ids
        blinded_ids = self._blind_set(ids, random_factors)
        self.messenger.send(blinded_ids)
        self.logger.log("Passive party finished sending blinded ids to active party.")

        # 3. unblind then hash signed ids
        signed_blined_ids = self.messenger.recv()
        signed_ids = self._unblind_set(signed_blined_ids, random_factors)
        hashed_signed_ids = PassiveRSAPSI._hash_set(signed_ids)
        self.messenger.send(hashed_signed_ids)
        self.logger.log(
            "Passive party finished sending hashed signed ids to active party"
        )

        # 4. receive intersection
        intersection_hashed_ids = set(self.messenger.recv())
        intersections = []
        for idx, hash_val in enumerate(hashed_signed_ids):
            if hash_val in intersection_hashed_ids:
                intersections.append(ids[idx])

        self.logger.log("Size of intersection: {}".format(len(intersections)))
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

    def run_offline(self, ids: List[int]) -> None:
        print("[PASSIVE] Start the offline protocol...")
        n_elements = len(ids)
        begin = time.time()
        randoms = [randbelow(self.LARGEST_RANDOM) for _ in range(n_elements)]
        print("Generating random numbers time: {:.5f}".format(time.time() - begin))
        target_dir = os.path.join(Path.home(), ".linkefl")
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        full_path = os.path.join(target_dir, self.RANDOMS_FILENAME)
        with open(full_path, "wb") as f:
            pickle.dump(randoms, f)
        print("[PASSIVE] Finish the offline protocol.")

    def run_online(self, ids: List[int]) -> List[int]:
        start_time = time.time()
        public_key = self._sync_pubkey()
        self.cryptosystem = PartialRSA(raw_public_key=public_key)

        # generate random factors and blind ids
        begin = time.time()
        full_path = os.path.join(Path.home(), ".linkefl", self.RANDOMS_FILENAME)
        with open(full_path, "rb") as f:
            randoms = pickle.load(f)
        random_factors = self._random_factors_mp_pool(
            randoms, num_workers=self.num_workers
        )
        print("Only random factors time: {:.5f}".format(time.time() - begin))
        begin = time.time()
        blinded_ids = self._blind_set(ids, random_factors)
        print("Only blind set time: {:.5f}".format(time.time() - begin))
        self.messenger.send(blinded_ids)

        signed_blined_ids = self.messenger.recv()
        begin = time.time()
        signed_ids = self._unblind_set(signed_blined_ids, random_factors)
        hashed_signed_ids = PassiveRSAPSI._hash_set(signed_ids)
        print("Unblind set and hash set time: {:.5f}".format(time.time() - begin))
        self.messenger.send(hashed_signed_ids)

        os.remove(full_path)
        intersection_hashed_ids = set(self.messenger.recv())
        intersections = []
        for idx, hash_val in enumerate(hashed_signed_ids):
            if hash_val in intersection_hashed_ids:
                intersections.append(ids[idx])
        print(colored("Size of intersection: {}".format(len(intersections)), "red"))
        print("Total time: {}".format(time.time() - start_time))

        return intersections

    def _sync_pubkey(self):
        self.logger.log("[PASSIVE] Requesting RSA public key...")
        self.messenger.send(Const.START_SIGNAL)
        n, e = self.messenger.recv()
        pub_key = RSA.construct((n, e))
        self.logger.log("[PASSIVE] Receive RSA public key successfully.")
        return pub_key

    def _random_factors(self, randoms):
        random_factors = []
        for r in randoms:
            r_inv = self.cryptosystem.inverse(r)
            r_encrypted = self.cryptosystem.encrypt(r)
            random_factors.append((r_inv, r_encrypted))

        return random_factors

    def _random_factors_mp_pool(self, randoms, num_workers=1):
        e = self.cryptosystem.pub_key.e
        n = self.cryptosystem.pub_key.n
        if num_workers == 1:
            return [_target_mp_pool(r, e, n) for r in randoms]

        with multiprocessing.Pool(num_workers) as p:
            random_factors = p.map(
                functools.partial(_target_mp_pool, e=e, n=n), randoms
            )

        return random_factors

    def _random_factors_thread(self, randoms, n_threads=-1):
        if n_threads == -1:
            n_threads = os.cpu_count()

        n = self.cryptosystem.pub_key.n
        r_invs = [gmpy2.invert(r, n) for r in randoms]
        r_encs = self.cryptosystem.encrypt_vector(randoms, num_workers=n_threads)
        random_factors = list(zip(r_invs, r_encs))

        return random_factors

    def _blind(self, x, rf, n):
        assert 0 <= x < n, f"x should be in range [0, {n})"
        rf_encrypted = rf[1]
        return self.cryptosystem.mulmod(x, rf_encrypted, n)

    def _unblind(self, x, rf, n):
        assert 0 <= x < n, f"x should be in range [0, {n})"
        rf_inv = rf[0]
        return self.cryptosystem.mulmod(x, rf_inv, n)

    def _blind_set(self, X, random_factors):
        blinded_set = []
        n = gmpy2.mpz(self.cryptosystem.pub_key.n)
        for x, rf in zip(X, random_factors):
            # Remember to pass n as an argument to self._blind instead of
            # obtaining it within self._blind. This can save A LOT OF time.
            b = self._blind(x, rf, n)
            blinded_set.append(b)

        return blinded_set

    def _unblind_set(self, X, random_factors):
        unblinded_set = []
        n = gmpy2.mpz(self.cryptosystem.pub_key.n)
        for x, rf in zip(X, random_factors):
            # Remember to pass n as an argument to self._unblind instead of
            # obtaining it within self._blind. This can save A LOT OF time when
            # set size is large.
            u = self._unblind(x, rf, n)
            unblinded_set.append(u)

        return unblinded_set

    @staticmethod
    def _hash_set(signed_set):
        return [hashlib.sha256(str(item).encode()).hexdigest() for item in signed_set]


if __name__ == "__main__":
    import argparse

    from linkefl.common.factory import logger_factory
    from linkefl.dataio import gen_dummy_ids
    from linkefl.messenger import FastSocket

    #   Option 1: split the protocol
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str)
    args = parser.parse_args()

    # # 1. Get sample IDs
    # _ids = gen_dummy_ids(size=100000, option=Const.SEQUENCE)
    #
    # # 2. Initialize messenger
    # _messenger = FastSocket(role=Const.PASSIVE_NAME,
    #                         active_ip='127.0.0.1',
    #                         active_port=20000,
    #                         passive_ip='127.0.0.1',
    #                         passive_port=30000)
    # _logger = logger_factory(role=Const.ACTIVE_NAME)
    #
    # # 3. Start the RSA-Blind-Signature protocol
    # alice = PassiveRSAPSI(_messenger, _logger)
    # if args.phase == 'offline':
    #     alice.run_offline(_ids)
    # elif args.phase == 'online':
    #     alice.run_online(_ids)
    # else:
    #     raise ValueError(f"command line argument `--phase` can only"
    #                      f"take `offline` and `online`, "
    #                      f"but {args.phase} got instead")
    #
    # # 4. close messenger
    # _messenger.close()

    # '''
    #  Option 2: run the whole protocol
    # 1. Get sample IDs
    _ids = gen_dummy_ids(size=100000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20000,
        passive_ip="127.0.0.1",
        passive_port=30000,
    )
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    # 3. Start the RSA-Blind-Signature protocol
    passive_party = PassiveRSAPSI(
        messenger=_messenger, logger=_logger, num_workers=os.cpu_count()
    )
    intersections_ = passive_party.run(_ids)
    print(len(intersections_))

    # 4. Close messenger
    _messenger.close()
    # '''
