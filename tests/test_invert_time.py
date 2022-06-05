"""the conlusion from this code snippit is that gmpy2.invert is quite efficient"""

import random
import time

import gmpy2
from phe import paillier


pub_key, priv_key = paillier.generate_paillier_keypair(n_length=1024)
n_squared = pub_key.n ** 2

size = 100000
residues= [random.uniform(-1, 1) for _ in range(size)]
enc_residues = [pub_key.encrypt(r) for r in residues]

start_time = time.time()
print('started...')
for r in enc_residues:
    r_invert = gmpy2.invert(r.ciphertext(False), n_squared)
print('elapsed time: {}'.format(time.time() - start_time))
