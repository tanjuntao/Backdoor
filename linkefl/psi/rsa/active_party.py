import argparse
import hashlib
import os
from pathlib import Path
import pickle
import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import RSACrypto
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket


class RSAPSIActive:
    def __init__(self, ids, messenger, cryptosystem, logger, num_workers=-1):
        self.ids = ids
        self.messenger = messenger
        self.cryptosystem = cryptosystem
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.HASHED_IDS_FILENAME = 'hashed_signed_ids.pkl'
        self.HERE = os.path.abspath(os.path.dirname(__file__))
        self.logger = logger

    def _send_pub_key(self):
        signal = self.messenger.recv()
        if signal == Const.START_SIGNAL:
            n = self.cryptosystem.pub_key.n
            e = self.cryptosystem.pub_key.e
            self.messenger.send([n, e])
            self.logger.log('[ACTIVE] Finish sending RSA public key.')
        else:
            raise ValueError('Invalid start signal.')

    @staticmethod
    def _hash_set(signed_set):
        return [hashlib.sha256(str(item).encode()).hexdigest()
                for item in signed_set]

    def _intersect(self, active_hashed_ids, passive_hashed_ids):
        intersections = []
        # directly convert Python list to Python set via the inbuild set() function is
        # faster than initilizing an empty set and then adding items sequentially
        # to this set.
        passive_hashed_set = set(passive_hashed_ids)
        for idx, hash_val in enumerate(active_hashed_ids):
            if hash_val in passive_hashed_set:
                intersections.append(self.ids[idx])

        # Before this function returns, the Python GC will delete the passive_hashed_set,
        # which is REALLY time-consuming if the set size is big, e.g, >= 40 million.
        return intersections

    def run_offline(self):
        print('[ACTIVE] Start the offline protocol...')
        begin = time.time()
        signed_ids = self.cryptosystem.sign_set_thread(self.ids, n_threads=self.num_workers)
        print('Signing self id set time: {:.5f}'.format(time.time() - begin))
        hashed_signed_ids = RSAPSIActive._hash_set(signed_ids)
        del signed_ids # save momory

        target_dir = os.path.join(Path.home(), '.linkefl')
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        full_path = os.path.join(target_dir, self.HASHED_IDS_FILENAME)
        with open(full_path, 'wb') as f:
            pickle.dump(hashed_signed_ids, f)
        print('[ACTIVE] Finish the offline protocol.')

    def run_online(self):
        print('[ACTIVE] Started the online protocol, listening...')
        self._send_pub_key()

        blinded_ids = self.messenger.recv()
        begin = time.time()
        signed_blinded_ids = self.cryptosystem.sign_set_thread(blinded_ids,
                                                               n_threads=self.num_workers)
        print('Signing passive id set time: {:.5f}'.format(time.time() - begin))
        self.messenger.send(signed_blinded_ids)

        passive_hashed_signed_ids = self.messenger.recv()
        full_path = os.path.join(Path.home(), '.linkefl', self.HASHED_IDS_FILENAME)
        with open(full_path, 'rb') as f:
            active_hashed_signed_ids = pickle.load(f)

        begin = time.time()
        intersections = self._intersect(active_hashed_signed_ids, passive_hashed_signed_ids)
        del active_hashed_signed_ids  # save memory
        del passive_hashed_signed_ids # save memory
        print('Intersection time: {}'.format(time.time() - begin))
        print(colored('Size of intersection: {}'.format(len(intersections)), 'red'))
        self.messenger.send(intersections)

        os.remove(full_path)
        print('[ACTIVE] Finish the online protocol.')

        return intersections

    def run(self):
        # 1. sync RSA public key
        self.logger.log('Active party starts PSI, listening...')
        self._send_pub_key()
        start = time.time()

        # 2. signing blinded ids of passive party
        blinded_ids = self.messenger.recv()
        signed_blinded_ids = self.cryptosystem.sign_set_thread(
            blinded_ids,
            n_threads=self.num_workers
        )
        self.messenger.send(signed_blinded_ids)
        self.logger.log('Active party sends signed blinded ids back to passive party.')
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.RUNNING,
            begin=start,
            end=None,
            duration=time.time() - start,
            progress=0.4
        )

        # 3. signing and hashing its own ids
        signed_ids = self.cryptosystem.sign_set_thread(self.ids,
                                                       n_threads=self.num_workers)
        active_hashed_signed_ids = RSAPSIActive._hash_set(signed_ids)
        self.logger.log('Active party finished signing and hashing its own ids.')
        self.logger.log_component(
            name=Const.RSA_PSI,
            status=Const.RUNNING,
            begin=start,
            end=None,
            duration=time.time() - start,
            progress=0.8
        )

        # 4. receiving hashed signed ids from passive party
        passive_hashed_signed_ids = self.messenger.recv()

        # 5. find the intersection
        intersections = self._intersect(active_hashed_signed_ids, passive_hashed_signed_ids)
        del active_hashed_signed_ids  # save momory
        del passive_hashed_signed_ids # save momory
        self.logger.log('Size of intersection: {}'.format(len(intersections)))
        self.messenger.send(intersections)

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
    #     _crypto = RSACrypto()
    #     bob = RSAPSIActive(_ids, _messenger, _crypto, _logger)
    #     bob.run_offline()
    #
    # elif args.phase == 'online':
    #     _crypto = RSACrypto.from_private()
    #     bob = RSAPSIActive(_ids, _messenger, _crypto, _logger)
    #     bob.run_online()
    #
    # else:
    #     raise ValueError(f"command line argument `--phase` can only"
    #                      f"take `offline` and `online`, "
    #                      f"but {args.phase} got instead")
    #
    # # 4. close messenger
    # _messenger.close()

    # '''
    ######   Option 2: run the whole protocol   ######
    # 1. get sample IDs
    _ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocket(role=Const.ACTIVE_NAME,
                            active_ip='127.0.0.1',
                            active_port=20000,
                            passive_ip='127.0.0.1',
                            passive_port=30000)
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    # 3. Initialize cryptosystem
    _crypto = RSACrypto()

    # 4. Start the RSA-Blind-Signature protocol
    active_party = RSAPSIActive(_ids, _messenger, _crypto, _logger)
    intersections_ = active_party.run()

    # 5. Close messenger
    _messenger.close()
    # '''
