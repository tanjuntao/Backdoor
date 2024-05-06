"""
Apply on the output of the last layer of bottom model
"""

import time

import numpy as np

from linkefl.crypto import Paillier
from linkefl.crypto.paillier import encode

crypto = Paillier()


if __name__ == "__main__":
    precision = 0.001
    batch_size = 128
    output_dim = 100  # change to 100 for cifar100
    total_time = 0

    ouput_embedding = np.random.rand(batch_size, output_dim)
    weight = np.random.rand(output_dim, output_dim)

    start = time.time()
    enc_output_embedding = np.array(
        crypto.encrypt_vector(ouput_embedding.flatten())
    ).reshape(batch_size, output_dim)
    weight, encode_mappings = encode(weight, crypto.pub_key, precision=precision)
    res = np.matmul(enc_output_embedding, weight)
    total_time += (time.time() - start) * 2

    start = time.time()
    grad, _ = encode(ouput_embedding, crypto.pub_key, precision=precision)
    weight_grad = np.matmul(enc_output_embedding.transpose(), grad)
    total_time += time.time() - start

    print(f"elapsed time: {total_time}")
