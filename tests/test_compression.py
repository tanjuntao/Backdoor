import bz2
import gzip
import lzma
import pickle
import random
import sys
import time

import blosc
import mgzip

from linkefl.crypto import Paillier


if __name__ == '__main__':
    paillier = Paillier(key_size=1024)
    num_enc_values = 100000
    random_list = [random.random() for _ in range(num_enc_values)]
    print('start encryption...')
    enc_random_list = [paillier.encrypt(val) for val in random_list]
    print('done.')

    with open('no_compression.pickle', 'wb') as f:
        pickle.dump(enc_random_list, f)

    with gzip.open("gzip_test.gz", "wb") as f:
        pickle.dump(enc_random_list, f)

    with bz2.BZ2File('bz2_test.pbz2', 'wb') as f:
        pickle.dump(enc_random_list, f)

    with lzma.open("lmza_test.xz", "wb") as f:
        pickle.dump(enc_random_list, f)

    with mgzip.open('mgzip_test.mgz', 'wb') as f:
        pickle.dump(enc_random_list, f)

    start = time.time()
    raw_pickled = pickle.dumps(enc_random_list)
    print('raw pickle time: {}'.format(time.time() - start))
    start = time.time()
    blosc_pickled = blosc.compress(pickle.dumps(enc_random_list))
    with open("blosc_test.blosc", "wb") as f:
        f.write(blosc_pickled)

    print('blosc pickle time: {}'.format(time.time() - start))
    print('raw pickled size: {}'.format(sys.getsizeof(raw_pickled)))
    print('blosc pickled size: {}'.format(sys.getsizeof(blosc_pickled)))



