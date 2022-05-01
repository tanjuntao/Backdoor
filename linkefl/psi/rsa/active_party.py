import argparse
import hashlib
import os
import pickle
import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.crypto import RSACrypto
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocketMessenger


class RSAPSIActive:
    def __init__(self, ids, messenger, cryptosystem, num_workers=-1):
        self.ids = ids
        self.messenger = messenger
        self.cryptosystem = cryptosystem
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.HASHED_IDS_FILENAME = '.hashed_ids.pkl'
        self.HERE = os.path.abspath(os.path.dirname(__file__))

    def _send_pub_key(self):
        signal = self.messenger.recv()
        if signal == Const.START_SIGNAL:
            n = self.cryptosystem.pub_key.n
            e = self.cryptosystem.pub_key.e
            self.messenger.send([n, e])
            print('[ACTIVE] Finish sending public key.')
        else:
            raise ValueError('Invalid start signal.')

    def _hash_set(self, signed_set):
        return [hashlib.sha256(str(item).encode()).hexdigest()
                for item in signed_set]

    def _intersect(self, active_hashed_ids, passive_hashed_ids):
        intersections = []
        passive_hashed_set = set(passive_hashed_ids)
        for idx, hash_val in enumerate(active_hashed_ids):
            if hash_val in passive_hashed_set:
                intersections.append(self.ids[idx])

        return intersections

    def run_offline(self):
        print('[ACTIVE] Start the offline protocol...')
        begin = time.time()
        signed_ids = self.cryptosystem.sign_set_thread(self.ids, n_threads=self.num_workers)
        print('Signing self id set time: {:.5f}'.format(time.time() - begin))
        hashed_ids = self._hash_set(signed_ids)
        with open('{}/{}'.format(self.HERE, self.HASHED_IDS_FILENAME), 'wb') as f:
            pickle.dump(hashed_ids, f)
        print('[ACTIVE] Finish the offline protocol.')

    def run_online(self):
        print('[ACTIVE] Started the online protocol, listening...')
        self._send_pub_key()

        blinded_ids = self.messenger.recv()
        begin = time.time()
        blinded_signed_ids = self.cryptosystem.sign_set_thread(blinded_ids,
                                                               n_threads=self.num_workers)
        print('Signing passive id set time: {:.5f}'.format(time.time() - begin))
        self.messenger.send(blinded_signed_ids)

        passive_hashed_ids = self.messenger.recv()
        with open('{}/{}'.format(self.HERE, self.HASHED_IDS_FILENAME), 'rb') as f:
            active_hashed_ids = pickle.load(f)

        begin = time.time()
        intersections = self._intersect(active_hashed_ids, passive_hashed_ids)
        print('Intersection time: {}'.format(time.time() - begin))
        print(colored('Size of intersection: {}'.format(len(intersections)), 'red'))
        self.messenger.send(intersections)

        os.remove('{}/{}'.format(self.HERE, self.HASHED_IDS_FILENAME))
        print('[ACTIVE] Finish the online protocol.')

        return intersections


if __name__ == '__main__':
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str)
    args = parser.parse_args()

    # 1. get sample IDs
    _ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocketMessenger(role=Const.ACTIVE_NAME,
                                     active_ip='127.0.0.1',
                                     active_port=20000,
                                     passive_ip='127.0.0.1',
                                     passive_port=30000)

    # 3. Start the RSA-Blind-Signature protocol
    if args.phase == 'offline':
        _crypto = RSACrypto()
        bob = RSAPSIActive(_ids, _messenger, _crypto)
        bob.run_offline()

    elif args.phase == 'online':
        _crypto = RSACrypto.from_private()
        bob = RSAPSIActive(_ids, _messenger, _crypto)
        bob.run_online()

    else:
        raise ValueError(f"command line argument `--phase` can only"
                         f"take `offline` and `online`, "
                         f"but {args.phase} got instead")

    # 4. close messenger
    _messenger.close()