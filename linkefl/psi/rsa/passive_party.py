import argparse
import functools
import hashlib
import multiprocessing
import os
from pathlib import Path
import pickle
from secrets import randbelow
import time

from Crypto.PublicKey import RSA
import gmpy2
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import PartialRSACrypto
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket


def _target_mp_pool(r, e, n):
    r_inv = gmpy2.invert(r, n)
    r_encrypted = gmpy2.powmod(r, e, n)

    return r_inv, r_encrypted


class RSAPSIPassive:
    def __init__(self, ids, messenger, num_workers=-1):
        self.ids = ids
        self.messenger = messenger
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.RANDOMS_FILENAME = 'randoms.pkl'
        self.LARGEST_RANDOM = pow(2, 512)
        self.HERE = os.path.abspath(os.path.dirname(__file__))
        self.logger = logger_factory(role=Const.PASSIVE_NAME)

    def _get_pub_key(self):
        self.logger.log('[PASSIVE] Requesting RSA public key...')
        self.messenger.send(Const.START_SIGNAL)
        n, e = self.messenger.recv()
        pub_key = RSA.construct((n, e))
        self.cryptosystem = PartialRSACrypto(pub_key=pub_key)
        self.logger.log('[PASSIVE] Receive RSA public key successfully.')

    def _random_factors(self, randoms):
        random_factors = []
        for r in randoms:
            r_inv = self.cryptosystem.inverse(r)
            r_encrypted = self.cryptosystem.encrypt(r)
            random_factors.append((r_inv, r_encrypted))

        return random_factors

    def _random_factors_mp_pool(self, randoms, n_processes=-1):
        if n_processes == -1:
            n_processes = os.cpu_count()

        e = self.cryptosystem.pub_key.e
        n = self.cryptosystem.pub_key.n

        with multiprocessing.Pool(n_processes) as p:
            random_factors = p.map(functools.partial(_target_mp_pool, e=e, n=n), randoms)

        return random_factors

    def _random_factors_thread(self, randoms, n_threads=-1):
        if n_threads == -1:
            n_threads = os.cpu_count()

        n = self.cryptosystem.pub_key.n
        r_invs = [gmpy2.invert(r, n) for r in randoms]
        r_encs = self.cryptosystem.encrypt_set_thread(randoms, n_threads=n_threads)
        random_factors = list(zip(r_invs, r_encs))

        return random_factors

    def _blind(self, x, rf, n):
        assert  0 <= x < n, f"x should be in range [0, {n})"
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
        return [hashlib.sha256(str(item).encode()).hexdigest()
                for item in signed_set]

    def run_offline(self):
        print('[PASSIVE] Start the offline protocol...')
        n_elements = len(self.ids)
        begin = time.time()
        randoms = [randbelow(self.LARGEST_RANDOM) for _ in range(n_elements)]
        print('Generating random numbers time: {:.5f}'.format(time.time() - begin))
        target_dir = os.path.join(Path.home(), '.linkefl')
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        full_path = os.path.join(target_dir, self.RANDOMS_FILENAME)
        with open(full_path, 'wb') as f:
            pickle.dump(randoms, f)
        print('[PASSIVE] Finish the offline protocol.')

    def run_online(self):
        start_time = time.time()
        self._get_pub_key()

        # generate random factors and blind ids
        begin = time.time()
        full_path = os.path.join(Path.home(), '.linkefl', self.RANDOMS_FILENAME)
        with open(full_path, 'rb') as f:
            randoms = pickle.load(f)
        random_factors = self._random_factors_mp_pool(randoms,
                                                      n_processes=self.num_workers)
        print('Only random factors time: {:.5f}'.format(time.time() - begin))
        begin = time.time()
        blinded_ids = self._blind_set(self.ids, random_factors)
        print('Only blind set time: {:.5f}'.format(time.time() - begin))
        self.messenger.send(blinded_ids)

        signed_blined_ids = self.messenger.recv()
        begin = time.time()
        signed_ids = self._unblind_set(signed_blined_ids, random_factors)
        hashed_signed_ids = RSAPSIPassive._hash_set(signed_ids)
        print('Unblind set and hash set time: {:.5f}'.format(time.time() - begin))
        self.messenger.send(hashed_signed_ids)

        os.remove(full_path)
        intersection = self.messenger.recv()
        print(colored('Size of intersection: {}'.format(len(intersection)), 'red'))
        print('Total time: {}'.format(time.time() - start_time))

        return intersection

    def run(self):
        # sync RSA public key
        self._get_pub_key()
        start = time.time()

        # 1. generate random factors
        randoms = [randbelow(self.LARGEST_RANDOM) for _ in range(len(self.ids))]
        random_factors = self._random_factors_mp_pool(randoms,
                                                      n_processes=self.num_workers)
        self.logger.log('Passive party finished genrating random factors.')
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.RUNNING,
            begin=start,
            end=None,
            duration=time.time() - start,
            progress=0.1
        )

        # 2. blind ids
        blinded_ids = self._blind_set(self.ids, random_factors)
        self.messenger.send(blinded_ids)
        self.logger.log('Passive party finished sending blinded ids to active party.')
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.RUNNING,
            begin=start,
            end=None,
            duration=time.time() - start,
            progress=0.2
        )

        # 3. unblind then hash signed ids
        signed_blined_ids = self.messenger.recv()
        signed_ids = self._unblind_set(signed_blined_ids, random_factors)
        hashed_signed_ids = RSAPSIPassive._hash_set(signed_ids)
        self.messenger.send(hashed_signed_ids)
        self.logger.log('Passive party finished sending hashed signed ids to active party')
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.RUNNING,
            begin=start,
            end=None,
            duration=time.time() - start,
            progress=0.6
        )

        # 4. receive intersection
        intersections = self.messenger.recv()
        self.logger.log('Size of intersection: {}'.format(len(intersections)))

        self.logger.log('Total protocol execution time: {:.5f}'.format(time.time() - start))
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.SUCCESS,
            begin=start,
            end=time.time(),
            duration=time.time() - start,
            progress=1.0
        )
        return intersections


if __name__ == '__main__':
    ######   Option 1: split the protocol   ######
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str)
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
    #
    # # 3. Start the RSA-Blind-Signature protocol
    # alice = RSAPSIPassive(_ids, _messenger)
    # if args.phase == 'offline':
    #     alice.run_offline()
    # elif args.phase == 'online':
    #     alice.run_online()
    # else:
    #     raise ValueError(f"command line argument `--phase` can only"
    #                      f"take `offline` and `online`, "
    #                      f"but {args.phase} got instead")
    #
    # # 4. close messenger
    # _messenger.close()

    # '''
    ######   Option 2: run the whole protocol   ######
    # 1. Get sample IDs
    _ids = gen_dummy_ids(size=100000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocket(role=Const.PASSIVE_NAME,
                            active_ip='127.0.0.1',
                            active_port=20000,
                            passive_ip='127.0.0.1',
                            passive_port=30000)

    # 3. Start the RSA-Blind-Signature protocol
    passive_party = RSAPSIPassive(_ids, _messenger)
    intersections = passive_party.run()

    # 4. Close messenger
    _messenger.close()
    # '''

