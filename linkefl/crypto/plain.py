from .base import CryptoSystem, PartialCryptoSystem

import numpy as np
import torch


class PartialPlain(PartialCryptoSystem):
    def __init__(self, pub_key=None):
        super(PartialPlain, self).__init__(pub_key)

    def encrypt(self, plaintext):
        return plaintext

    def encrypt_vector(self, plain_vector, using_mp=False, n_processes=None):
        if type(plain_vector) == np.ndarray:
            return list(plain_vector)
        elif type(plain_vector) == torch.Tensor:
            return list(plain_vector.numpy())
        else:
            return plain_vector.copy() # Python list


class Plain(CryptoSystem):
    """Pseudo cryptosystem."""
    def __init__(self, key_size=0):
        super(Plain, self).__init__(key_size)
        self.pub_key, self.priv_key = self._gen_key(key_size)

    def _gen_key(self, key_size):
        return None, None

    def encrypt(self, plaintext):
        return plaintext

    def decrypt(self, ciphertext):
        return ciphertext

    def encrypt_vector(self, plain_vector, using_mp=False, n_processes=None):
        if type(plain_vector) == np.ndarray:
            return list(plain_vector)
        elif type(plain_vector) == torch.Tensor:
            return list(plain_vector.numpy())
        else:
            return plain_vector.copy() # Python list

    def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
        return [val for val in cipher_vector]