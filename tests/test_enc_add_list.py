import concurrent.futures
import os
import time

import gmpy2
import numpy as np
from phe import EncryptedNumber

from linkefl.crypto.paillier import FastPaillier, fast_cipher_sum

if __name__ == "__main__":
    size = 10000
    BASE = 16
    np.random.seed(0)
    crypto = FastPaillier(key_size=1024, num_enc_zeros=1000, gen_from_set=False)
    random_data = np.random.rand(size) * 2 - 1
    enc_data = crypto.encrypt_vector(random_data)

    # option 1: compute the summation of encrypted numbers with the buildin sum function
    start = time.time()
    res = sum(enc_data)
    print("python sum time: {}".format(time.time() - start))
    print(crypto.decrypt(res))

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
        res = gmpy2.mod(gmpy2.mul(res, cipher), crypto.pub_key.nsquare)
    final_sum_enc = EncryptedNumber(crypto.pub_key, int(res), min_exponent)
    print("manually sum time: {}".format(time.time() - start_time))
    # print('Mulmod time: {}'.format(time.time() - start))
    print(crypto.decrypt(final_sum_enc))

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
    for exp, ciphers in exp_2_encs.items():
        print(f"exp: {exp}, length of ciphers: {len(ciphers)}")

    def target_func(ciphers, curr_exp, min_exp, base, nsquare):
        gmpy2.get_context().allow_release_gil = True
        new_exp = pow(base, curr_exp - min_exp)
        return gmpy2.powmod_base_list(ciphers, new_exp, nsquare)

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
                    crypto.pub_key.nsquare,
                )
            )
        for task in concurrent.futures.as_completed(tasks):
            aligned_ciphers = task.result()
            res = aligned_ciphers[0]
            for cipher in aligned_ciphers[1:]:
                res = gmpy2.mod(gmpy2.mul(res, cipher), crypto.pub_key.nsquare)
            sumed_encs.append(res)
        final_sum_enc = sumed_encs[0]
        for cipher in sumed_encs[1:]:
            final_sum_enc = gmpy2.mod(
                gmpy2.mul(final_sum_enc, cipher), crypto.pub_key.nsquare
            )
    print("thread pool time: {}".format(time.time() - start_time))
    print(
        crypto.decrypt(
            EncryptedNumber(crypto.pub_key, int(final_sum_enc), min_exponent)
        )
    )

    # option 4: use LinkeFL's fast_add_ciphers
    start_time = time.time()
    res = fast_cipher_sum(cipher_vector=enc_data)
    print("fast_cipher_sum time: {}".format(time.time() - start_time))
    print(crypto.decrypt(res))
