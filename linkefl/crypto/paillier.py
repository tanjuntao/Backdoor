import math
import random
import os

import gmpy2
import numpy as np
from phe import paillier
from phe import EncryptedNumber
import torch

from .base import CryptoSystem, PartialCryptoSystem
from linkefl.config import BaseConfig


def _cal_enc_zeros(public_key, num_enc_zeros, gen_from_set):
    """Calculate pre-computed encrypted zeros for faster encryption later on"""
    def _subset(_base_set, _public_key, _n_squared):
        _backtrack(0, [], _base_set, _public_key, _n_squared)

    def _backtrack(start_idx, path, _base_set, _public_key, _n_squared):
        # save current state
        if path:
            ciphertext = gmpy2.mpz(1)
            for val in path:
                ciphertext = gmpy2.mul(ciphertext, val) % _n_squared
            enc_zero = EncryptedNumber(_public_key, int(ciphertext), 0)
            enc_zeros.append(enc_zero)

        # dfs
        for i in range(start_idx, len(_base_set)):
            path.append(_base_set[i])
            _backtrack(i + 1, path, _base_set, _public_key, _n_squared)
            path.pop()

    # driven code
    n_squared = public_key.n ** 2
    if not gen_from_set:
        enc_zeros = [public_key.encrypt(0) for _ in range(num_enc_zeros)]
    else:
        num_entries = math.ceil(math.log(num_enc_zeros, 2))
        base_set = []
        for _ in range(num_entries):
            r = random.SystemRandom().randrange(1, public_key.n)
            r_pow = gmpy2.powmod(r, public_key.n, n_squared)
            base_set.append(r_pow)
        enc_zeros = []
        _subset(base_set, public_key, n_squared)

    return enc_zeros


# TODO: add support for torch datatype when do encryption
class PartialPaillier(PartialCryptoSystem):
    def __init__(self, pub_key):
        super(PartialPaillier, self).__init__(pub_key)

    def encrypt(self, plaintext):
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)
        return self.pub_key.encrypt(plaintext)

    def encrypt_vector(self, plain_vector, using_mp=False, n_processes=None):
        return [self.encrypt(val) for val in plain_vector]


class PartialFastPaillier(PartialCryptoSystem):
    def __init__(self, pub_key, num_enc_zeros=10000, gen_from_set=True):
        super(PartialFastPaillier, self).__init__(pub_key)
        self._enc_zeros = _cal_enc_zeros(self.pub_key, num_enc_zeros, gen_from_set)

    def encrypt(self, plaintext):
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)
        enc_zero = random.choice(self._enc_zeros)
        return enc_zero + plaintext

    def encrypt_vector(self, plain_vector, using_mp=False, n_processes=None):
        return [self.encrypt(val) for val in plain_vector]


class Paillier(CryptoSystem):
    """Paillier additive homomorphic cryptosystem.

    Paillier cryptosystem satisfies additive homomorphism,
    which means that:

    - Enc(u) + Enc(v) = Enc(u+v) and Dec(Enc(u+v)) = u+v
    - Enc(u) * v = Enc(u*v) and Dec(Enc(u*v)) = u*v

    *note*: "+" and "*" in the above equations are not arithmetic symbols,
    but notations to represent addition and multiplication operations on
    encrypted data.
    """
    def __init__(self, key_size=1024):
        super(Paillier, self).__init__(key_size)
        self.pub_key, self.priv_key = self._gen_key(key_size)

    @classmethod
    def from_config(cls, config):
        assert isinstance(config, BaseConfig), 'config object should be an ' \
                                               'instance of BaseConfig class.'
        return cls(key_size=config.KEY_SIZE)

    def _gen_key(self, key_size):
        pub_key, priv_key = paillier.generate_paillier_keypair(n_length=key_size)
        return pub_key, priv_key

    def encrypt(self, plaintext):
        # TODO: add support for PyTorch tensor data type
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)
        return self.pub_key.encrypt(plaintext)

    def decrypt(self, ciphertext):
        if isinstance(ciphertext, EncryptedNumber):
            return self.priv_key.decrypt(ciphertext)
        else:
            return ciphertext

    def encrypt_vector(self, plain_vector, using_mp=False, n_processes=None):
        if n_processes is None:
            n_processes = os.cpu_count()
        self.params_checking(plain_vector, using_mp, n_processes)

        if type(plain_vector) == np.ndarray:
            plain_vector = list(plain_vector)
        if type(plain_vector) == torch.Tensor:
            plain_vector = list(plain_vector.numpy())

        if not using_mp:
            return [self.encrypt(val) for val in plain_vector]
        else:
            return self.func_mp(self.encrypt, plain_vector, n_processes)

    def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
        if n_processes is None:
            n_processes = os.cpu_count()
        self.params_checking(cipher_vector, using_mp, n_processes)

        if not using_mp:
            return [self.decrypt(cipher) for cipher in cipher_vector]
        else:
            return self.func_mp(self.decrypt, cipher_vector, n_processes)


class FastPaillier(CryptoSystem):
    """
    Faster paillier encryption using pre-computed encrypted zeros.
    """
    def __init__(self, key_size=1024, num_enc_zeros=10000, gen_from_set=True):
        """
        Initialize a FastPaillier instance

        Args:
            key_size: key size of paillier cryptosystem
            num_enc_zeros: how many encrypted zeros to generate offline
            gen_from_set: whether to generate an encrypted zero from a finite
              set or using paillier encryption function directly.

        Returns:
            None
        """
        # TODO: move the process of generating encrypted zeros to offline phase
        super(FastPaillier, self).__init__(key_size)
        self.num_enc_zeros = num_enc_zeros
        self.gen_from_set = gen_from_set

        self.pub_key, self.priv_key = self._gen_key(key_size)
        self.n_squared = self.pub_key.n ** 2

        print('Generating encrypted zeros...')
        self._enc_zeros = _cal_enc_zeros(self.pub_key, num_enc_zeros, gen_from_set)
        print('Done!')

    @classmethod
    def from_config(cls, config):
        assert isinstance(config, BaseConfig), 'config object should be an ' \
                                               'instance of BaseConfig class.'
        return cls(key_size=config.KEY_SIZE,
                   num_enc_zeros=config.NUM_ENC_ZEROS,
                   gen_from_set=config.GEN_FROM_SET)

    def _gen_key(self, key_size):
        pub_key, priv_key = paillier.generate_paillier_keypair(n_length=key_size)
        return pub_key, priv_key

    def encrypt(self, plaintext):
        # TODO: load encrypted zeros from pickled file
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)
        # TODO: improve security by random choice without replacement
        enc_zero = random.choice(self._enc_zeros)
        return enc_zero + plaintext

    def decrypt(self, ciphertext):
        return self.priv_key.decrypt(ciphertext)

    def encrypt_vector(self, plain_vector, using_mp=False, n_processes=None):
        if n_processes is None:
            n_processes = os.cpu_count()
        self.params_checking(plain_vector, using_mp, n_processes)

        if type(plain_vector) == np.ndarray:
            plain_vector = list(plain_vector)
        if type(plain_vector) == torch.Tensor:
            plain_vector = list(plain_vector.numpy())

        if not using_mp:
            return [self.encrypt(val) for val in plain_vector]
        else:
            return self.func_mp(self.encrypt, plain_vector, n_processes)

    def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
        if n_processes is None:
            n_processes = os.cpu_count()
        self.params_checking(cipher_vector, using_mp, n_processes)

        if not using_mp:
            return [self.decrypt(cipher) for cipher in cipher_vector]
        else:
            return self.func_mp(self.decrypt, cipher_vector, n_processes)


# class PartialPaillier(Paillier):
#     def __init__(self, pub_key):
#         # it is just OK not calling super() constructor function in python subclass
#         # if we don't need to initilize the self variables
#         # and the methods in superclass are inherited automatically
#         self.pub_key = pub_key
#
#     def decrypt(self, ciphertext):
#         raise Exception('Trying to do decryption in a cryptosystem without '
#                         'private key.')
#
#     def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
#         raise Exception('Trying to do decryption in a cryptosystem without '
#                         'private key.')
#
#
# class PartialFastPaillier(FastPaillier):
#     def __init__(self, pub_key, num_enc_zeros=10000, gen_from_set=False):
#         self.pub_key = pub_key
#         self._enc_zeros = self.cal_enc_zeros(pub_key, num_enc_zeros, gen_from_set)
#
#     def decrypt(self, ciphertext):
#         raise Exception('Trying to do decryption in a cryptosystem without '
#                         'private key.')
#
#     def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
#         raise Exception('Trying to do decryption in a cryptosystem without '
#                         'private key.')


if __name__ == '__main__':
    # paillier = FastPaillier(num_enc_zeros=64, gen_from_set=True)
    # enc_zero_ = paillier._enc_zeros[20]
    # print(type(enc_zero_))
    # print(paillier.decrypt(enc_zero_ + 19))

    paillier = Paillier(key_size=1024)
    # paillier = FastPaillier(num_enc_zeros=128, gen_from_set=True)
    # paillier = Plain()
    item1 = 10
    item2 = 20.033
    print(paillier.decrypt(paillier.encrypt(item1)))
    print(paillier.decrypt(paillier.encrypt(item2)))

    # vector = [2.1, 3, 0.0, 0.3]
    # vector = np.arange(2000)
    # vector = np.array([0, 1, 2, 3])
    # vector = list(vector)
    # vector = torch.Tensor([0, 1, 2, 3])
    vector = torch.arange(10000)
    # vector = np.random.rand(100000)
    import time

    print(vector[:10])
    start = time.time()
    enc_vector = paillier.encrypt_vector(plain_vector=vector, using_mp=True)
    dec_vector = paillier.decrypt_vector(enc_vector, using_mp=True)
    print(dec_vector[:10])
    print('Elapsed time: {:.5f}'.format(time.time() - start))

