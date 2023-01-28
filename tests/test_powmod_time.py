"""the conclusion from this code snippit is that gmpy2.powmod is efficient
when the exponent argument is small, e.g., less than 5000
"""

import random
import time

import gmpy2
from phe import paillier

pub_key, priv_key = paillier.generate_paillier_keypair(n_length=1024)
n_squared = pub_key.n**2

size = 100000
exps = [random.randint(100, 30000) for _ in range(size)]
residues = [random.uniform(-1, 1) for _ in range(10)]
enc_residues = [pub_key.encrypt(r) for r in residues]

start_time = time.time()
print("started...")
for ex in exps:
    res = gmpy2.powmod(enc_residues[0].ciphertext(False), ex, n_squared)
print("elapsed time: {}".format(time.time() - start_time))
