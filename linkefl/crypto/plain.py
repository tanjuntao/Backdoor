from typing import Any, Union

import numpy as np
import torch

from linkefl.base import BaseCryptoSystem, BasePartialCryptoSystem
from linkefl.common.const import Const


class PartialPlain(BasePartialCryptoSystem):
    def __init__(self, raw_public_key):
        super(PartialPlain, self).__init__()
        self.pub_key = raw_public_key
        self.type: str = Const.PLAIN

    def encrypt(self, plaintext: Any) -> Any:
        return plaintext

    def encrypt_vector(
        self,
        plain_vector: Union[list, np.ndarray, torch.Tensor],
        pool=None,
        num_workers: int = 1,
    ) -> list:
        if type(plain_vector) == list:
            return plain_vector.copy()
        elif type(plain_vector) == np.ndarray:
            return list(plain_vector)
        elif type(plain_vector) == torch.Tensor:
            return list(plain_vector.numpy())
        else:
            raise TypeError(
                "Only Python list, Numpy Array, and PyTorch Tensor can be passed to"
                f" this method, but got {type(plain_vector)} instead."
            )


class Plain(BaseCryptoSystem):
    """Pseudo cryptosystem."""

    def __init__(self, key_size: int = 0):
        super(Plain, self).__init__(key_size)
        # this line takes no effect, just for API consistency
        self.pub_key, self.priv_key = self._gen_key(key_size)
        self.type: str = Const.PLAIN

    def _gen_key(self, key_size):
        return None, None

    def encrypt(self, plaintext: Any) -> Any:
        return plaintext

    def decrypt(self, ciphertext: Any) -> Any:
        return ciphertext

    def encrypt_vector(
        self,
        plain_vector: Union[list, np.ndarray, torch.Tensor],
        pool=None,
        num_workers: int = 1,
    ) -> list:
        if type(plain_vector) == list:
            return plain_vector.copy()
        elif type(plain_vector) == np.ndarray:
            return list(plain_vector)
        elif type(plain_vector) == torch.Tensor:
            return list(plain_vector.numpy())
        else:
            raise TypeError(
                "Only Python list, Numpy Array, and PyTorch Tensor can be passed to"
                f" this method, but got {type(plain_vector)} instead."
            )

    def decrypt_vector(
        self,
        cipher_vector: Union[list, np.ndarray],
        pool=None,
        num_workers: int = 1,
    ) -> list:
        assert type(cipher_vector) in (
            list,
            np.ndarray,
        ), (
            "cipher_vector's dtype can only be Python list or Numpy array, but got"
            f" {type(cipher_vector)} instead."
        )
        return [val for val in cipher_vector]
