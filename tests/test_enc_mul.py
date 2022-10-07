"""This testing script shows that when do Paillier ciphertext multipliation,
we should transform the negtive encoding to its inverse, which can accelarate
the computation a lot.
"""

import random
import time

import gmpy2
from phe import generate_paillier_keypair, EncodedNumber


if __name__ == '__main__':
    pub_key, priv_key = generate_paillier_keypair(n_length=1024)
    n = pub_key.n
    nsquare = pub_key.nsquare
    residue = 0.123456
    enc_residue = pub_key.encrypt(residue)
    ciphertext = enc_residue.ciphertext(False)
    nec_cipher = int(gmpy2.invert(ciphertext, nsquare))

    size = 1000
    raw_data = [-1 * random.random() for _ in range(size)]
    encoded_data = [EncodedNumber.encode(pub_key, val) for val in raw_data]

    # option 1: slow
    start_time = time.time()
    for encoding in encoded_data:
        gmpy2.powmod(ciphertext, encoding.encoding, nsquare)
    print('naive powmod time: {}'.format(time.time() - start_time))

    # option 2: MUCH faster
    start_time = time.time()
    for encoding in encoded_data:
        # nec_cipher = int(gmpy2.invert(ciphertext, nsquare))
        neg_encoding = n - encoding.encoding
        gmpy2.powmod(nec_cipher, neg_encoding, nsquare)
    print('neg cipher time: {}'.format(time.time() - start_time))
