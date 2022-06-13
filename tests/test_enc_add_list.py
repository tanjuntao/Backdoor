import os
import pickle
import random
import time
from phe import paillier


pub_key, priv_key = paillier.generate_paillier_keypair(n_length=1024)
size = 1000000

if not os.path.exists('./enc_data.pkl'):
    print('start encryption...')
    data = [random.randint(0, 1000_0000) / 10_0000 for _ in range(size)]
    enc_data = [pub_key.encrypt(val) for val in data]
    with open('./enc_data.pkl', 'wb') as f:
        pickle.dump(enc_data, f)
    print('done.')
else:
    with open('./enc_data.pkl', 'rb') as f:
        enc_data = pickle.load(f)

sorted_enc_data = sorted(enc_data, key=lambda enc_val: enc_val.exponent)

print(sorted_enc_data[0].exponent)
print(sorted_enc_data[-1].exponent)

# option 1
start = time.time()
res = sum(enc_data)
print('Elapsed time: {}'.format(time.time() - start))

# option 2
exponents = [enc_val.exponent for enc_val in enc_data]



