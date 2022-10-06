import multiprocessing
import multiprocessing.pool
import os
import random
import time

import gmpy2
from line_profiler_pycharm import profile
from phe import paillier, EncodedNumber
from phe.util import mulmod
from tqdm import tqdm, trange


@profile
def dec_vector(cipher_vector, private_key):
    return [private_key.decrypt(cipher) for cipher in cipher_vector]


def dec_vector_thread(cipher_vector, private_key, thread_pool):
    ciphertexts = [encrypted_number.ciphertext(be_secure=False)
                   for encrypted_number in cipher_vector]
    exponents = [encrypted_number.exponent for encrypted_number in cipher_vector]
    n_workers = thread_pool._processes
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
        result = thread_pool.apply_async(_target_dec_thread,
                                         args=(ciphertexts, exponents, private_key, start, end))
        async_results.append(result)
    for idx, result in enumerate(async_results):
        assert result.get() is True, "worker process did not finish " \
                                     "within default timeout"
    return ciphertexts

def _target_dec_thread(ciphertexts, exponents, private_key, start, end):
    p, psquare, hp = private_key.p, private_key.psquare, private_key.hp
    q, qsquare, hq = private_key.q, private_key.qsquare, private_key.hq
    powmod_p = gmpy2.powmod_list(ciphertexts[start:end], p - 1, psquare)
    powmod_q = gmpy2.powmod_list(ciphertexts[start:end], q - 1, qsquare)
    public_key = private_key.public_key
    sublist_idx = 0
    for k in range(start, end):
        decrypt_to_p = mulmod(private_key.l_function(powmod_p[sublist_idx], p), hp, p)
        decrypt_to_q = mulmod(private_key.l_function(powmod_q[sublist_idx], q), hq, q)
        encoding = private_key.crt(decrypt_to_p, decrypt_to_q)
        encoding = EncodedNumber(public_key, encoding, exponents[k])
        ciphertexts[k] = encoding.decode()
        sublist_idx += 1
    return True


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
        result = process_pool.apply_async(_target_dec_pool,
                                          args=(shared_data, private_key, start, end))
        async_results.append(result)
    for idx, result in enumerate(async_results):
        assert result.get() is True, "worker process did not finish " \
                                     "within default timeout"

    return [item for item in shared_data]


def _target_dec_pool(ciphertexts, private_key, start, end):
    for k in range(start, end):
        ciphertexts[k] = private_key.decrypt(ciphertexts[k])
    return True


if __name__ == '__main__':
    pub_key, priv_key = paillier.generate_paillier_keypair(n_length=1024)
    size = 100
    random_data = [random.random() for _ in range(size)]
    print(random_data[-1])
    print('start encryption...')
    enc_data = []
    for rand in tqdm(random_data):
        enc_data.append(pub_key.encrypt(rand))
    print('done.')

    # single process
    start_time = time.time()
    dec_result = dec_vector(cipher_vector=enc_data, private_key=priv_key)
    print('single process elapsed time: {}'.format(time.time() - start_time))

    # multiple threads
    start_time = time.time()
    thread_pool = multiprocessing.pool.ThreadPool(os.cpu_count())
    print('thread pool creation time: {}'.format(time.time() - start_time))
    start_time = time.time()
    dec_result_threadpool = dec_vector_thread(
        enc_data,
        priv_key,
        thread_pool
    )
    print('multiple threads elapsed time: {}'.format(time.time() - start_time))
    print(dec_result_threadpool[-1])
    thread_pool.close()

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
    print(dec_result_pool[-1])
    pool.close()

