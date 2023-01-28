import concurrent.futures
import multiprocessing
import multiprocessing.pool
import os
import pickle
import random
import time

import gmpy2
from phe import EncodedNumber, EncryptedNumber, paillier
from tqdm import tqdm

from linkefl.crypto import fast_add_ciphers

if __name__ == "__main__":
    size = 10000
    BASE = 16
    pub_key, priv_key = paillier.generate_paillier_keypair(n_length=1024)
    random_data = [random.random() for _ in range(size)]
    enc_data = []
    for rand in tqdm(random_data):
        enc_data.append(pub_key.encrypt(rand))

    sorted_enc_data = sorted(enc_data, key=lambda enc_val: enc_val.exponent)
    print(sorted_enc_data[0].exponent)
    print(sorted_enc_data[-1].exponent)

    # option 1: compute the summation of encrypted numbers with the buildin sum function
    start = time.time()
    res = sum(enc_data)
    print("Elapsed time: {}".format(time.time() - start))
    print(priv_key.decrypt(res))

    # option 2: break down the whole computation
    start_time = time.time()
    exponents = [enc_val.exponent for enc_val in enc_data]
    min_exponent = min(exponents)
    aligned_enc_data = [
        enc_val * pow(BASE, enc_val.exponent - min_exponent) for enc_val in enc_data
    ]
    ciphertexts = [gmpy2.mpz(enc_val.ciphertext(False)) for enc_val in aligned_enc_data]
    start = time.time()
    res = ciphertexts[0]
    for cipher in ciphertexts[1:]:
        res = gmpy2.mod(gmpy2.mul(res, cipher), pub_key.nsquare)
    final_sum_enc = EncryptedNumber(pub_key, int(res), min_exponent)
    print("Elapsed time: {}".format(time.time() - start_time))
    # print('Mulmod time: {}'.format(time.time() - start))
    print(priv_key.decrypt(final_sum_enc))

    # option 3: using powmod_list() function
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
    start_time = time.time()

    exp_2_encs = {}
    for enc_val in enc_data:
        if enc_val.exponent not in exp_2_encs:
            exp_2_encs[enc_val.exponent] = [gmpy2.mpz(enc_val.ciphertext(False))]
        else:
            exp_2_encs[enc_val.exponent].append(gmpy2.mpz(enc_val.ciphertext(False)))
    min_exponent = min(exp_2_encs.keys())

    def target_func(ciphers, curr_exp, min_exp, base, nsquare):
        gmpy2.get_context().allow_release_gil = True
        new_exp = pow(base, curr_exp - min_exp)
        return gmpy2.powmod_list(ciphers, new_exp, nsquare)

    with executor:
        tasks = []
        sumed_encs = []
        for exp in exp_2_encs.keys():
            tasks.append(
                executor.submit(
                    target_func,
                    exp_2_encs[exp],
                    exp,
                    min_exponent,
                    BASE,
                    pub_key.nsquare,
                )
            )
        for task in concurrent.futures.as_completed(tasks):
            aligned_ciphers = task.result()
            res = aligned_ciphers[0]
            for cipher in aligned_ciphers[1:]:
                res = gmpy2.mod(gmpy2.mul(res, cipher), pub_key.nsquare)
            sumed_encs.append(res)
        final_sum_enc = sumed_encs[0]
        for cipher in sumed_encs[1:]:
            final_sum_enc = gmpy2.mod(gmpy2.mul(final_sum_enc, cipher), pub_key.nsquare)
    print("Elapsed time: {}".format(time.time() - start_time))
    print(priv_key.decrypt(EncryptedNumber(pub_key, int(final_sum_enc), min_exponent)))

    thread_pool = multiprocessing.pool.ThreadPool(os.cpu_count())
    start_time = time.time()
    res = fast_add_ciphers(cipher_vector=enc_data, thread_pool=thread_pool)
    print("ThreadPool time: {}".format(time.time() - start_time))
    print(priv_key.decrypt(res))
