import bz2
import gzip
import lzma
import pickle
import random
import sys
import time
import zlib

import blosc
import mgzip
import numpy as np

from linkefl.crypto import Paillier


def is_pubkey_same(pub_key, size=10):
    random_vector = [random.random() for _ in range(size)]
    enc_vector = [pub_key.encrypt(val) for val in random_vector]
    print(id(enc_vector[0].public_key), id(enc_vector[1].public_key))

    pickle_dumps = pickle.dumps(enc_vector)
    pickle_loads = pickle.loads(pickle_dumps)
    print(id(pickle_loads[0].public_key), id(pickle_loads[1].public_key))


if __name__ == "__main__":
    paillier = Paillier(key_size=1024)
    num_enc_values = int(1e5)
    random_list = [random.random() for _ in range(num_enc_values)]
    print("start encryption...")

    # [Python list] zlib is better, pickle:blosc:zlib = 900364:900380:795105
    # enc_random_list = random_list

    # [Paillier EncryptedNumber] zlib is better,
    # pickle:blosc:zlib = 28703315:28703331:25923663
    enc_random_list = [paillier.encrypt(val) for val in random_list]

    # [Paillier EncryptedNumber ciphertext alone], zlib is better
    # pickle:blosc:zlib = 26105602:26105618:25900541
    # enc_random_list = [paillier.encrypt(val) for val in random_list]
    # ciphers = [cipher.ciphertext(False) for cipher in enc_random_list]
    # enc_random_list = ciphers

    # [np float] blosc is better, pickle:blosc:zlib = 800195:124571:594805
    # enc_random_list = np.linspace(0, 101, num_enc_values)

    # [np random float] blosc is better, pickle:blosc:zlib = 800195:700662:754582
    # enc_random_list = np.random.rand(num_enc_values)

    # [np int] blosc is better, pickle:blosc:zlib = 800195:103050:140409
    # enc_random_list = np.random.randint(0, 100, size=num_enc_values)
    print("done.")

    # raw pickle
    start = time.time()
    raw_pickled = pickle.dumps(enc_random_list)
    print("raw pickle time: {}".format(time.time() - start))

    # blosc
    start = time.time()
    blosc_pickled = blosc.compress(pickle.dumps(enc_random_list))
    # with open("blosc_test.blosc", "wb") as f:
    #     f.write(blosc_pickled)
    print("blosc pickle time: {}".format(time.time() - start))

    # Python buildin zlib
    start = time.time()
    zlib_pickled = zlib.compress(pickle.dumps(enc_random_list))
    print("raw zlib pickle time: {}".format(time.time() - start))

    # binary data size
    print("raw pickled size: {}".format(sys.getsizeof(raw_pickled)))
    print("blosc pickled size: {}".format(sys.getsizeof(blosc_pickled)))
    print("zlib pickled size: {}".format(sys.getsizeof(zlib_pickled)))

    # with open('no_compression.pickle', 'wb') as f:
    #     pickle.dump(enc_random_list, f)
    #
    # with gzip.open("gzip_test.gz", "wb") as f:
    #     pickle.dump(enc_random_list, f)
    #
    # with bz2.BZ2File('bz2_test.pbz2', 'wb') as f:
    #     pickle.dump(enc_random_list, f)
    #
    # with lzma.open("lmza_test.xz", "wb") as f:
    #     pickle.dump(enc_random_list, f)
    #
    # with mgzip.open('mgzip_test.mgz', 'wb') as f:
    #     pickle.dump(enc_random_list, f)
