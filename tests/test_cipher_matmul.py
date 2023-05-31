import multiprocessing
import time

import numpy as np

from linkefl.crypto import FastPaillier
from linkefl.crypto.paillier import encode, fast_cipher_matmul

if __name__ == "__main__":
    shape = 64
    enc_mat_shape = (shape, shape)
    plain_mat_shape = (shape, shape)
    # enc_mat_shape = (32, 128)
    # plain_mat_shape = (128, 64)
    precision = 0.001
    n_workers = 8
    np.random.seed(0)

    crypto = FastPaillier()
    enc_matrix = np.random.rand(*enc_mat_shape) * 2 - 1
    enc_matrix = np.array(crypto.encrypt_vector(enc_matrix.flatten())).reshape(
        enc_mat_shape
    )
    plain_matrix = np.random.rand(*plain_mat_shape) * 2 - 1
    encode_matrix, encode_mappings = encode(
        plain_matrix, crypto.pub_key, precision=precision
    )

    start_time = time.time()
    res = np.matmul(enc_matrix, plain_matrix)
    print(crypto.decrypt_vector(res[0][:4]))
    print(f"plain matmul: {time.time() - start_time}")

    start_time = time.time()
    res = np.matmul(enc_matrix, encode_matrix)
    print(crypto.decrypt_vector(res[0][:4]))
    print(f"encode matmul: {time.time() - start_time}")

    scheduler_pool = multiprocessing.pool.ThreadPool(n_workers)
    executor_pool = multiprocessing.pool.ThreadPool(n_workers)

    start_time = time.time()
    res = fast_cipher_matmul(
        cipher_matrix=enc_matrix,
        encode_matrix=encode_matrix,
        encode_mappings=encode_mappings,
        executor_pool=executor_pool,
        scheduler_pool=scheduler_pool,
    )
    print(crypto.decrypt_vector(res[0][:4]))
    print(f"fast matmul: {time.time() - start_time}")

    scheduler_pool.close()
    executor_pool.close()
