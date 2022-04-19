from .base import CryptoSystem

import numpy as np
import torch


class Plain(CryptoSystem):
    """Pseudo cryptosystem."""
    def __init__(self, key_size=None):
        super(Plain, self).__init__(key_size)
        self.pub_key, self.priv_key = self.__gen_key()

    def __gen_key(self):
        return None, None

    def encrypt(self, plaintext):
        return plaintext

    def decrypt(self, ciphertext):
        return ciphertext

    def encrypt_vector(self, plain_vector, using_mp=False, n_processes=None):
        if type(plain_vector) == np.ndarray:
            plain_vector = list(plain_vector)
        if type(plain_vector) == torch.Tensor:
            plain_vector = list(plain_vector.numpy())

        return [val for val in plain_vector]

    def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
        return [val for val in cipher_vector]
