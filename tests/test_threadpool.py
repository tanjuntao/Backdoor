import multiprocessing
import time
from multiprocessing.pool import Pool, ThreadPool
import os
import random

import gmpy2
from phe import paillier, EncodedNumber, EncryptedNumber
from phe.util import mulmod


def target_enc_pool(plaintexts, pkey, start, end):
    # option 1: call paillier encryption interface directly
    # for k in range(start, end):
    #     plaintexts[k] = pkey.encrypt(plaintexts[k])
    # return True

    # option 2: manually do encryption
    n = pkey.n
    nsquare = pkey.nsquare
    for k in range(start, end):
        encoding = EncodedNumber.encode(pkey, plaintexts[k])
        nude_ciphertext = (1 + n * encoding.encoding) % nsquare
        r_value = random.SystemRandom().randrange(1, n)
        obfuscator = gmpy2.powmod(r_value, n, nsquare)
        ciphertext = mulmod(nude_ciphertext, obfuscator, nsquare)
        encrypted_number = EncryptedNumber(pkey, ciphertext, encoding.exponent)
        encrypted_number._EncryptedNumber__is_obfuscated = True
        plaintexts[k] = encrypted_number
    return True


def target_dec_pool(ciphertexts, skey, start, end):
    for k in range(start, end):
        ciphertexts[k] = skey.decrypt(ciphertexts[k])
    return True


def target_enc_thread(plaintexts, pkey, start, end):
    # option 1: call paillier encryption interface directly
    # for k in range(start, end):
    #     plaintexts[k] = pkey.encrypt(plaintexts[k])
    # return True

    n = pkey.n
    nsquare = pkey.nsquare
    r_values = [random.SystemRandom().randrange(1, n) for _ in range(end - start)]
    obfuscators = gmpy2.powmod_list(r_values, n, nsquare)

    r_idx = 0
    for k in range(start, end):
        encoding = EncodedNumber.encode(pkey, plaintexts[k])
        nude_ciphertext = (1 + n * encoding.encoding) % nsquare
        ciphertext = mulmod(nude_ciphertext, obfuscators[r_idx], nsquare)
        encrypted_number = EncryptedNumber(pkey, ciphertext, encoding.exponent)
        encrypted_number._EncryptedNumber__is_obfuscated = True
        plaintexts[k] = encrypted_number
        r_idx += 1
    return True



if __name__ == '__main__':
    pub_key, priv_key = paillier.generate_paillier_keypair(n_length=1024)

    n_workers = os.cpu_count()
    mp_pool = Pool(n_workers)
    thread_pool = ThreadPool(n_workers)
    epochs = 5
    data_size = 100_00
    quotient = data_size // n_workers
    remainder = data_size % n_workers

    # multiprocessing.Pool
    pool_start = time.time()
    for i in range(epochs):
        print(f"epoch: {i}")
        random.seed(i)
        data = [random.random() for _ in range(data_size)]
        print(data[-1])
        manager = multiprocessing.Manager()
        shared_data = manager.list(data)
        async_results = []
        for idx in range(n_workers):
            start_ = idx * quotient
            end_ = (idx + 1) * quotient
            if idx == n_workers - 1:
                end_ += remainder
            result = mp_pool.apply_async(target_enc_pool, args=(shared_data, pub_key, start_, end_))
            async_results.append(result)

        for idx, result in enumerate(async_results):
            assert result.get() is True, "worker process did not finish " \
                                         "within default timeout"
            if idx == len(async_results) - 1:
                print(priv_key.decrypt(shared_data[-1]))

        print(f"epoch {i} finished.\n")
        # time.sleep(1)
    pool_end = time.time()

    # exit(0)

    # multiprocessing.ThreadPool
    thread_start = time.time()
    for i in range(epochs):
        print(f"epoch: {i}")
        random.seed(i)
        data = [random.random() for _ in range(data_size)]
        print(data[-1])
        async_results = []
        for idx in range(n_workers):
            start_ = idx * quotient
            end_ = (idx + 1) * quotient
            if idx == n_workers - 1:
                end_ += remainder
            result = thread_pool.apply_async(target_enc_thread, args=(data, pub_key, start_, end_))
            async_results.append(result)

        for idx, result in enumerate(async_results):
            assert result.get() is True, "worker process did not finish " \
                                         "within default timeout"
            if idx == len(async_results) - 1:
                print(priv_key.decrypt(data[-1]))

        print(f"epoch {i} finished.\n")
        # time.sleep(1)
    thread_end = time.time()


    # test if pool can be applied by different target function
    pool_start = time.time()
    for i in range(epochs):
        print(f"epoch: {i}")
        random.seed(i)
        data = [random.random() for _ in range(data_size)]
        print(data[-1])
        manager = multiprocessing.Manager()
        shared_data = manager.list(data)
        async_results = []

        # apply encryption
        for idx in range(n_workers):
            start_ = idx * quotient
            end_ = (idx + 1) * quotient
            if idx == n_workers - 1:
                end_ += remainder
            result = mp_pool.apply_async(target_enc_pool, args=(shared_data, pub_key, start_, end_))
            async_results.append(result)
        for idx, result in enumerate(async_results):
            assert result.get() is True, "worker process did not finish " \
                                         "within default timeout"
            if idx == len(async_results) - 1:
                print(priv_key.decrypt(shared_data[-1]))

        # apply decryption
        for idx in range(n_workers):
            start_ = idx * quotient
            end_ = (idx + 1) * quotient
            if idx == n_workers - 1:
                end_ += remainder
            result = mp_pool.apply_async(target_dec_pool, args=(shared_data, priv_key, start_, end_))
            async_results.append(result)
        for idx, result in enumerate(async_results):
            assert result.get() is True, "worker process did not finish " \
                                         "within default timeout"
            if idx == len(async_results) - 1:
                print(shared_data[-1])

        print(f"epoch {i} finished.\n")
        # time.sleep(1)
    pool_end = time.time()


    mp_pool.close()
    thread_pool.close()
    mp_pool.join()
    thread_pool.join()

    print(f"Pool elapsed time: {pool_end - pool_start}")
    print(f"Thread elapsed time: {thread_end - thread_start}")