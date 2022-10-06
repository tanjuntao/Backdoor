import copy
import multiprocessing.pool
import random
import time

import gmpy2
from line_profiler_pycharm import profile
from phe import paillier, EncodedNumber, EncryptedNumber
from tqdm import tqdm, trange


# @profile
def fast_enc_mul_list(plain_vector, enc_number, thread_pool):
    public_key = enc_number.public_key
    n = public_key.n
    max_int = public_key.max_int
    nsquare = public_key.nsquare
    ciphertext = enc_number.ciphertext(be_secure=False)
    exponent = enc_number.exponent
    ciphertext_inverse = gmpy2.invert(ciphertext, nsquare)

    pos_idxs, neg_idxs = [], []
    pos_exps, neg_exps = [], []
    for i, encoded_number in enumerate(plain_vector):
        encoding = encoded_number.encoding
        if n - max_int <= encoding:
            neg_idxs.append(i)
            neg_exps.append(n - encoding)
        else:
            pos_idxs.append(i)
            pos_exps.append(encoding)

    # slow
    # pos_res = gmpy2.powmod_exp_list(ciphertext, pos_exps, nsquare)
    # neg_res = gmpy2.powmod_exp_list(ciphertext_inverse, neg_exps, nsquare)

    # fast
    result_vector = [None] * len(plain_vector)
    async_results = []
    async_results.append(thread_pool.apply_async(_target_enc_mul_list, args=(ciphertext, pos_exps, nsquare)))
    async_results.append(thread_pool.apply_async(_target_enc_mul_list, args=(ciphertext_inverse, neg_exps, nsquare)))
    pos_res = async_results[0].get()
    neg_res = async_results[1].get()

    for pos_i, pos_cipher in zip(pos_idxs, pos_res):
        exp = plain_vector[pos_i].exponent + exponent
        result_vector[pos_i] = EncryptedNumber(public_key, int(pos_cipher), exp)
    for neg_i, neg_cipher in zip(neg_idxs, neg_res):
        exp = plain_vector[neg_i].exponent + exponent
        result_vector[neg_i] = EncryptedNumber(public_key, int(neg_cipher), exp)

    return result_vector


def _target_enc_mul_list(base, exps, nsquare):
    return gmpy2.powmod_exp_list(base, exps, nsquare)


def enc_mul_list(plain_vector, enc_number):
    return [val * enc_number for val in plain_vector]


if __name__ == '__main__':
    pub_key, priv_key = paillier.generate_paillier_keypair(n_length=1024)
    size = 100000
    epochs = 10
    precision = 0.0001
    residue = 0.123456
    enc_residue = pub_key.encrypt(residue)
    random_data = [random.random() - 0.5 for _ in range(size)] # offset be 0.5
    print(residue * random_data[-1])
    encoded_data = [EncodedNumber.encode(pub_key, val, precision) for val in tqdm(random_data)]

    # option 1
    start_time = time.time()
    for _ in trange(epochs):
        result = enc_mul_list(encoded_data, enc_residue)
    print('Elapsed time: {}'.format(time.time() - start_time))
    print(priv_key.decrypt(result[-1]))

    # option 2
    pool = multiprocessing.pool.ThreadPool(8)
    start_time = time.time()
    for _ in trange(epochs):
        fast_result = fast_enc_mul_list(encoded_data, enc_residue, pool)
    print('Elapsed time: {}'.format(time.time() - start_time))
    print(priv_key.decrypt(fast_result[-1]))
    pool.close()