"""
apply on the first conv layer
"""
import time

import numpy as np

from linkefl.crypto import Paillier
from linkefl.crypto.paillier import encode

crypto = Paillier()


if __name__ == "__main__":
    precision = 0.001
    conv_flat = np.random.rand(3 * 3 * 3)
    enc_conv_flat = np.array(crypto.encrypt_vector(conv_flat))
    input_data = np.random.rand(3 * 3 * 3, 1)
    input_data, encode_mappings = encode(
        input_data, crypto.pub_key, precision=precision
    )

    start = time.time()
    for i in range(100):
        single_conv_op = (input_data.flatten() * enc_conv_flat).sum()
        res = np.random.rand(1) - single_conv_op
        res = crypto.decrypt(res)
    single_conv_time = (time.time() - start) / 100

    conv_layer_time = single_conv_time * 28 * 28 * 64

    all_time = conv_layer_time * 4
    print(f"elapsed time: {all_time}")

    # cifar10 and vgg13 share the same first conv layer
