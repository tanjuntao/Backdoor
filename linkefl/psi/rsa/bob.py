import argparse
import hashlib
import os
import pickle
import time

from linkefl.config import PSIRSAConfig as Config
from linkefl.crypto import RSACrypto
from linkefl.dataio import get_ids
from linkefl.messenger import FastSocketMessenger


class Bob:
    def __init__(self, ids, messenger, cryptosystem):
        self.ids = ids
        self.messenger = messenger
        self.cryptosystem = cryptosystem
        self.hashed_id_file_name = '.hashed_ids.pkl'

    def send_pub_key(self):
        signal = self.messenger.recv()
        if signal == Config.START_SIGNAL:
            n = self.cryptosystem.pub_key.n
            e = self.cryptosystem.pub_key.e
            self.messenger.send([n, e])
            print('[Bob] Finish sending public key.')
        else:
            raise ValueError('Invalid start signal.')

    def hash_set(self, signed_set):
        return [hashlib.sha256(str(item).encode()).hexdigest()
                for item in signed_set]

    def run_offline(self):
        print('[Bob] Start the offline protocol...')
        begin = time.time()
        signed_ids = self.cryptosystem.sign_set_thread(self.ids)
        print('Signing bob id set time: {:.5f}'.format(time.time() - begin))
        hashed_ids = self.hash_set(signed_ids)
        with open('./{}'.format(self.hashed_id_file_name), 'wb') as f:
            pickle.dump(hashed_ids, f)
        print('[Bob] Finish the offline protocol.')

    def run_online(self):
        print('[Bob] Started the online protocol, listening...')
        self.send_pub_key()

        blinded_ids = self.messenger.recv()
        begin = time.time()
        blinded_signed_ids = self.cryptosystem.sign_set_thread(blinded_ids)
        print('Signing alice id set time: {:.5f}'.format(time.time() - begin))
        self.messenger.send(blinded_signed_ids)

        alice_hashed_ids = self.messenger.recv()
        with open('./{}'.format(self.hashed_id_file_name), 'rb') as f:
            bob_hashed_ids = pickle.load(f)

        begin = time.time()
        intersection = []
        alice_hashed_set = set(alice_hashed_ids)
        for i, hash_val in enumerate(bob_hashed_ids):
            if hash_val in alice_hashed_set:
                intersection.append(self.ids[i])
        print('Intersection time: {}'.format(time.time() - begin))
        print('Intersection size: {}'.format(len(intersection)))
        self.messenger.send(intersection)

        os.remove('./{}'.format(self.hashed_id_file_name))
        print('[Bob] Finish the online protocol.')


if __name__ == '__main__':
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str)
    args = parser.parse_args()

    # 1. get sample IDs
    _ids = get_ids(role=Config.ROLE_BOB)

    # 2. Initialize messenger
    _messenger = FastSocketMessenger(role=Config.ROLE_BOB)


    # 3. Start the RSA-Blind-Signature protocol
    if args.phase == 'offline':
        _crypto = RSACrypto()
        bob = Bob(_ids, _messenger, _crypto)
        bob.run_offline()

    elif args.phase == 'online':
        _crypto = RSACrypto.from_private()
        bob = Bob(_ids, _messenger, _crypto)
        bob.run_online()

    else:
        raise ValueError(f"command line argument `--phase` can only"
                         f"take `offline` and `online`, "
                         f"but {args.phase} got instead")

    # 4. close messenger
    _messenger.close()