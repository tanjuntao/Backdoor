"""
Apply on the output of the first layer
"""

import time

import numpy as np

from linkefl.crypto import Paillier

crypto = Paillier()


if __name__ == "__main__":
    conv_ouput_shape = (128, 64, 28, 28)
    data_a = np.random.rand(128 * 64)
    data_b = np.random.rand(128 * 64)

    start = time.time()
    for i in range(3):
        enc_data_a = np.array(crypto.encrypt_vector(data_a.flatten()))
        enc_data_b = np.array(crypto.encrypt_vector(data_b.flatten()))
        enc_res = enc_data_a + enc_data_b
        plain_res = crypto.decrypt_vector(enc_res)
    avg_time = (time.time() - start) / 3

    all_time = avg_time * 28 * 28
    print(f"elapsed time: {all_time}")
