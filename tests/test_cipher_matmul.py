import multiprocessing
import time

import numpy as np

from linkefl.crypto import FastPaillier
from linkefl.vfl.nn.enc_layer import ActiveEncLayer


if __name__ == '__main__':
    enc_mat_shape = (32, 128)
    plain_mat_shape = (128, 64)
    precision = 0.001
    n_workers = 8

    crypto = FastPaillier()
    enc_matrix = np.random.rand(*enc_mat_shape)
    enc_matrix = np.array(crypto.encrypt_vector(enc_matrix.flatten())).reshape(enc_mat_shape)
    plain_matrix = np.random.rand(*plain_mat_shape)
    plain_matrix = ActiveEncLayer._encode(plain_matrix, crypto.pub_key, precision=precision)

    scheduler_pool = multiprocessing.pool.ThreadPool(n_workers)
    executor_pool = multiprocessing.pool.ThreadPool(n_workers)

    start_time = time.time()
    ActiveEncLayer._cipher_matmul(
        cipher_matrix=enc_matrix,
        plain_matrix=plain_matrix,
        executor_pool=executor_pool,
        scheduler_pool=scheduler_pool
    )
    print('elapsed time: {}'.format(time.time() - start_time))

    scheduler_pool.close()
    executor_pool.close()

