import math
import random
import os

import gmpy2
import numpy as np
from phe import paillier
from phe import EncryptedNumber
import torch

from .base import CryptoSystem
from .plain import Plain


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
    def __init__(self, config, key_size=None):
        if key_size is None:
            key_size = config.DEFAULT_KEY_SIZE
        super(Paillier, self).__init__(key_size)
        self.config = config
        self.pub_key, self.priv_key = self.__gen_key()

    def __gen_key(self):
        pub_key, priv_key = paillier.generate_paillier_keypair(n_length=self.key_size)
        return pub_key, priv_key

    def encrypt(self, plaintext):
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)
        return self.pub_key.encrypt(plaintext)

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


class FastPaillier(CryptoSystem):
    """
    Faster paillier encryption using pre-computed encrypted zeros.
    """
    def __init__(self, config, key_size=None, num_enc_zeros=None, gen_from_set=None):
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
        if key_size is None:
            key_size = config.DEFAULT_KEY_SIZE
        if num_enc_zeros is None:
            num_enc_zeros = config.NUM_ENC_ZEROS
        if gen_from_set is None:
            gen_from_set = config.GEN_FROM_SET

        self.config = config
        super(FastPaillier, self).__init__(key_size)
        self.pub_key, self.priv_key = self.__gen_key()
        self.n_squared = self.pub_key.n ** 2
        self.gen_from_set = gen_from_set

        print('Generating encrypted zeros...')
        self._enc_zeros = self.cal_enc_zeros(self.pub_key, num_enc_zeros, gen_from_set)
        print('Done!')

    def __gen_key(self):
        pub_key, priv_key = paillier.generate_paillier_keypair(n_length=self.key_size)
        return pub_key, priv_key

    def cal_enc_zeros(self, public_key, num_enc_zeros, gen_from_set):
        """Calculate pre-computed encrypted zeros for later encryption."""
        def _subset(Base_set, Public_key, N_squared):
            _backtrack(0, [], Base_set, Public_key, N_squared)

        def _backtrack(start_idx, path, Base_set, Public_key, N_squared):
            # save current state
            if path:
                ciphertext = gmpy2.mpz(1)
                for val in path:
                    ciphertext = gmpy2.mul(ciphertext, val) % N_squared
                enc_zero = EncryptedNumber(Public_key, int(ciphertext), 0)
                enc_zeros.append(enc_zero)

            # dfs
            for i in range(start_idx, len(Base_set)):
                path.append(Base_set[i])
                _backtrack(i + 1, path, Base_set, Public_key, N_squared)
                path.pop()

        # Main function
        n_squared = public_key.n ** 2
        if not gen_from_set:
            enc_zeros = [public_key.encrypt(0) for _ in range(num_enc_zeros)]
        else:
            num_entries = int(math.log(num_enc_zeros, 2))
            base_set = []
            for _ in range(num_entries):
                r = random.SystemRandom().randrange(1, public_key.n)
                r_pow = gmpy2.powmod(r, public_key.n, n_squared)
                base_set.append(r_pow)
            enc_zeros = []
            _subset(base_set, public_key, n_squared)

        return enc_zeros

    def encrypt(self, plaintext):
        # TODO: load encrypted zeros from pickled file
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)
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


class PartialPlain(Plain):
    def __init__(self, pub_key, key_size=0):
        super(PartialPlain, self).__init__(key_size=key_size)
        self.pub_key = pub_key # overwritten

    def decrypt(self, ciphertext):
        raise Exception('Trying to do decryption in a cryptosystem without '
                        'private key.')

    def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
        raise Exception('Trying to do decryption in a cryptosystem without '
                        'private key.')


class PartialPaillier(Paillier):
    def __init__(self, config, pub_key, key_size=None):
        super(PartialPaillier, self).__init__(config, key_size=key_size)
        self.pub_key = pub_key # overwritten

    def decrypt(self, ciphertext):
        raise Exception('Trying to do decryption in a cryptosystem without '
                        'private key.')

    def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
        raise Exception('Trying to do decryption in a cryptosystem without '
                        'private key.')


class PartialFastPaillier(FastPaillier):
    def __init__(self,
                 config,
                 pub_key,
                 key_size=None,
                 num_enc_zeros=None,
                 gen_from_set=None):
        super(PartialFastPaillier, self).__init__(config,
                                                  key_size=key_size,
                                                  num_enc_zeros=num_enc_zeros,
                                                  gen_from_set=gen_from_set)
        # overwritten
        self.pub_key = pub_key
        # overwritten
        self._enc_zeros = self.cal_enc_zeros(self.pub_key, num_enc_zeros, gen_from_set)

    def decrypt(self, ciphertext):
        raise Exception('Trying to do decryption in a cryptosystem without '
                        'private key.')

    def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
        raise Exception('Trying to do decryption in a cryptosystem without '
                        'private key.')


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

