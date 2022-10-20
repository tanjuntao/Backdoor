from .base import CryptoSystem, PartialCryptoSystem

import numpy as np
import torch

from linkefl.common.const import Const


class PartialPlain(PartialCryptoSystem):
    def __init__(self, raw_public_key=None):
        super(PartialPlain, self).__init__()
        self.pub_key = raw_public_key
        self.type = Const.PLAIN

    def encrypt(self, plaintext):
        return plaintext

    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, pool=None):
        if type(plain_vector) == list:
            return plain_vector.copy()
        elif type(plain_vector) == np.ndarray:
            return list(plain_vector)
        elif type(plain_vector) == torch.Tensor:
            return list(plain_vector.numpy())
        else:
            raise TypeError("Only Python list, Numpy Array, and PyTorch Tensor can be"
                            " passed to this method.")


class Plain(CryptoSystem):
    """Pseudo cryptosystem."""
    def __init__(self, key_size=0):
        super(Plain, self).__init__(key_size)
        # this line takes no effect, just for API consistency
        self.pub_key, self.priv_key = self._gen_key(key_size)
        self.type = Const.PLAIN

    def _gen_key(self, key_size):
        return None, None

    def encrypt(self, plaintext):
        return plaintext

    def decrypt(self, ciphertext):
        return ciphertext

    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, pool=None):
        if type(plain_vector) == list:
            return plain_vector.copy()
        elif type(plain_vector) == np.ndarray:
            return list(plain_vector)
        elif type(plain_vector) == torch.Tensor:
            return list(plain_vector.numpy())
        else:
            raise TypeError("Only Python list, Numpy Array, and PyTorch Tensor can be"
                            " passed to this method.")

    def decrypt_vector(self, cipher_vector,
                       using_pool=False, n_workers=None, pool=None):
        assert type(cipher_vector) in (list, np.ndarray), \
            "cipher_vector's dtype can only be Python list or Numpy array."
        return [val for val in cipher_vector]