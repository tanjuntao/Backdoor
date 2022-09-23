import concurrent.futures
import multiprocessing
import os
import random
import time
import threading

import gmpy2
from phe import generate_paillier_keypair, EncodedNumber, EncryptedNumber
from phe.util import mulmod

from linkefl.crypto import Paillier


def target_func(randoms, start, end, n, n_square, release_gil=False):
    gmpy2.get_context().allow_release_gil = release_gil
    for i in range(start, end):
        randoms[i] = gmpy2.powmod(randoms[i], n, n_square)

    # powmod_list() will always release Python GIL, so release_gil flag has no effect
    # sub_randoms = randoms[start:end]
    # randoms[start:end] = gmpy2.powmod_list(sub_randoms, n, n_square)


def enc_vector(plain_vector, public_key, release_gil=False, n_threads=os.cpu_count()):
    size = len(plain_vector)
    n = public_key.n
    nsquare = public_key.nsquare
    randoms = [random.SystemRandom().randrange(1, n) for _ in range(size)]

    if n_threads == -1:
        n_threads = os.cpu_count()

    quotient = size // n_threads
    remainder = size % n_threads
    threads = []
    for idx in range(n_threads):
        start = idx * quotient
        end = (idx + 1) * quotient
        if idx == n_threads - 1:
            end += remainder
        t = threading.Thread(target=target_func,
                             args=(randoms, start, end, n, nsquare, release_gil))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    encrypted_vector = []
    for i, val in enumerate(plain_vector):
        encoding = EncodedNumber.encode(public_key, val)
        nude_ciphertext = (1 + n * encoding.encoding) % nsquare
        obfuscator = randoms[i]
        ciphertext = mulmod(nude_ciphertext, obfuscator, nsquare)
        encrypted_number = EncryptedNumber(public_key, ciphertext, encoding.exponent)
        encrypted_number._EncryptedNumber__is_obfuscated = True
        encrypted_vector.append(encrypted_number)

    return encrypted_vector


if __name__ == '__main__':
    size = 1200
    offset = 100
    paillier = Paillier(key_size=1024)
    pub_key, priv_key = paillier.pub_key, paillier.priv_key

    small_plaintexts = [pub_key.max_int - offset for _ in range(size)]
    # print(small_plaintexts[0])
    large_plaintexts = [-1 * (pub_key.max_int - offset) for _ in range(size)]

    start = time.time()
    result = [pub_key.encrypt(val) for val in small_plaintexts]
    print('small plaintexts encryption time: {}'.format(time.time() - start))

    # start = time.time()
    # resutl = [pub_key.encrypt(val) for val in large_plaintexts]
    # print('large plaintexts encryption time: {}'.format(time.time() - start))
    #
    # start_time = time.time()
    # ciphers = enc_vector(small_plaintexts, pub_key, release_gil=False)
    # print('single thread elapsed time: {}'.format(time.time() - start_time))
    # # print(priv_key.decrypt(ciphers[0]))

    start_time = time.time()
    ciphers = enc_vector(small_plaintexts, pub_key, release_gil=True)
    print('multi thread elapsed time: {}'.format(time.time() - start_time))

    start_time = time.time()
    pool = multiprocessing.Pool(os.cpu_count())
    ciphers = paillier.encrypt_data(small_plaintexts, pool)
    print('pool elapsed time: {}'.format(time.time() - start_time))

