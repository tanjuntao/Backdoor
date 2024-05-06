"""
Apply on the last layer of the bottom model
"""

import time

import numpy as np

from linkefl.crypto import Paillier
from linkefl.crypto.paillier import encode

crypto = Paillier()


if __name__ == "__main__":
    batch_size = 128
    output_dim = 100  # replace to 100 for cifar100
    precision = 0.001

    output_embedding = np.random.rand(batch_size, output_dim)
    weight = np.random.rand(output_dim, output_dim)

    start = time.time()
    for i in range(3):
        enc_weight = np.array(crypto.encrypt_vector(weight.flatten())).reshape(
            output_dim, output_dim
        )
        encode_embedding, encode_mappings = encode(
            output_embedding, crypto.pub_key, precision=precision
        )
        res = np.matmul(encode_embedding, enc_weight)
        noise = np.random.rand(batch_size, output_dim)
        noise = np.array(crypto.encrypt_vector(noise.flatten())).reshape(
            batch_size, output_dim
        )
        res = res - noise
        res = crypto.decrypt_vector(res.flatten())
    print(f"elapsed time: {(time.time() - start) / 3}")
