import multiprocessing
import multiprocessing.pool
import os
import random
import time

from phe import paillier
from tqdm import tqdm, trange


def dec_vector(cipher_vector, private_key):
    return [private_key.decrypt(cipher) for cipher in cipher_vector]


def dec_vector_pool(cipher_vector, private_key, n_workers=None, process_pool=None):
    if process_pool is None:
        if n_workers is None:
            n_workers = os.cpu_count()
        process_pool = multiprocessing.pool.Pool(n_workers)

    n_workers = process_pool._processes
    manager = multiprocessing.Manager()
    shared_data = manager.list(cipher_vector)
    data_size = len(cipher_vector)
    quotient = data_size // n_workers
    remainder = data_size % n_workers
    async_results = []
    for idx in range(n_workers):
        start = idx * quotient
        end = (idx + 1) * quotient
        if idx == n_workers - 1:
            end += remainder
        # target function will modify shared_data in-place
        result = process_pool.apply_async(_target_dec_vector,
                                          args=(shared_data, private_key, start, end))
        async_results.append(result)
    for idx, result in enumerate(async_results):
        assert result.get() is True, "worker process did not finish " \
                                     "within default timeout"

    return [item for item in shared_data]


def _target_dec_vector(ciphertexts, private_key, start, end):
    for k in range(start, end):
        ciphertexts[k] = private_key.decrypt(ciphertexts[k])
    return True


if __name__ == '__main__':
    pub_key, priv_key = paillier.generate_paillier_keypair(n_length=1024)
    size = 3000
    random_data = [random.random() for _ in range(size)]
    print('start encryption...')
    enc_data = []
    for rand in tqdm(random_data):
        enc_data.append(pub_key.encrypt(rand))
    print('done.')

    # single process
    start_time = time.time()
    dec_result = dec_vector(cipher_vector=enc_data, private_key=priv_key)
    print('single process elapsed time: {}'.format(time.time() - start_time))

    # multiple processes
    start_time = time.time()
    pool = multiprocessing.pool.Pool(os.cpu_count())
    print('pool creation time: {}'.format(time.time() - start_time))
    start_time = time.time()
    dec_result_pool = dec_vector_pool(
        enc_data,
        priv_key,
        process_pool=pool
    )
    print('multiple processes elapsed time: {}'.format(time.time() - start_time))
    pool.close()

