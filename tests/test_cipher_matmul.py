import multiprocessing
import time

import numpy as np

from linkefl.crypto import FastPaillier
from linkefl.crypto.paillier import encode, fast_cipher_matmul

if __name__ == "__main__":
    shape = 128
    # enc_mat_shape = (shape, shape)
    # plain_mat_shape = (shape, shape)
    enc_mat_shape = (32, 128)
    plain_mat_shape = (128, 64)
    precision = 0.001
    n_workers = 8

    crypto = FastPaillier()
    enc_matrix = np.random.rand(*enc_mat_shape) * 2 - 1
    enc_matrix = np.array(crypto.encrypt_vector(enc_matrix.flatten())).reshape(
        enc_mat_shape
    )
    plain_matrix = np.random.rand(*plain_mat_shape) * 2 - 1
    plain_matrix = encode(plain_matrix, crypto.pub_key, precision=precision)

    scheduler_pool = multiprocessing.pool.ThreadPool(n_workers)
    executor_pool = multiprocessing.pool.ThreadPool(n_workers)

    start_time = time.time()
    result_matrix = fast_cipher_matmul(
        cipher_matrix=enc_matrix,
        plain_matrix=plain_matrix,
        executor_pool=executor_pool,
        scheduler_pool=scheduler_pool,
    )
    print("elapsed time: {}".format(time.time() - start_time))

    scheduler_pool.close()
    executor_pool.close()
