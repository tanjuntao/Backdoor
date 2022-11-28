import copy
import functools
import multiprocessing
import os
import threading
from pathlib import Path

import gmpy2
from Crypto.PublicKey import RSA as CryptoRSA

from linkefl.base import BasePartialCryptoSystem, BaseCryptoSystem
from linkefl.common.const import Const
from linkefl.config import BaseConfig


class RSAPublicKey:
    def __init__(self, raw_pub_key):
        self.raw_pub_key = raw_pub_key
        self.n = raw_pub_key.n
        self.e = raw_pub_key.e

    def raw_encrypt(self, plaintext):
        return gmpy2.powmod(plaintext, self.e, self.n)

    def raw_encrypt_vector(self, plain_vector,
                           using_pool=False, n_workers=None, thread_pool=None):
        assert isinstance(plain_vector, list), \
            "in RSA cryptosystem, plain_vector can only be a Python list."

        if not using_pool:
            return [self.raw_encrypt(val) for val in plain_vector]

        if thread_pool is None:
            if n_workers is None:
                n_workers = os.cpu_count()
            thread_pool = multiprocessing.pool.ThreadPool(n_workers)

        # important: make a copy of original plain_vector, so it will not be modified
        plain_vector = copy.deepcopy(plain_vector)
        n_threads = thread_pool._processes
        e = self.e
        n = self.n
        quotient = len(plain_vector) // n_threads
        remainder = len(plain_vector) % n_threads
        threads = []
        for idx in range(n_threads):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_threads - 1:
                end += remainder
            # plain_vector will be modified in place
            t = threading.Thread(target=RSAPublicKey._target_enc_vector,
                                 args=(plain_vector, start, end, e, n))
            threads.append(t)
        for t in threads:
            t.join()

        return plain_vector

    @staticmethod
    def _target_enc_vector(plaintexts, start, end, e, n):
        sub_list = plaintexts[start:end]
        plaintexts[start:end] = gmpy2.powmod_list(sub_list, e, n)

    def raw_inverse(self, x):
        return gmpy2.invert(x, self.n)

    def raw_mulmod(self, x, y, z):
        # convert python integer to gmpy2.mpz, which can boost the computation
        if type(x) == int:
            x = gmpy2.mpz(x)
        if type(y) == int:
            y = gmpy2.mpz(y)
        if type(z) == int:
            z = gmpy2.mpz(z)
        return gmpy2.mod(gmpy2.mul(x, y), z)


class RSAPrivateKey:
    def __init__(self, raw_priv_key):
        self.raw_priv_key = raw_priv_key
        self.n = raw_priv_key.n
        self.d = raw_priv_key.d

    def raw_decrypt(self, ciphertext):
        return gmpy2.powmod(ciphertext, self.d, self.n)

    def raw_decrypt_vector(self, cipher_vector,
                           using_pool=False, n_workers=None, thread_pool=None):
        assert isinstance(cipher_vector, list), \
            "in RSA cryptosystem, cipher_vector can only be a Python list."

        if not using_pool:
            return [self.raw_decrypt(val) for val in cipher_vector]

        if thread_pool is None:
            if n_workers is None:
                n_workers = os.cpu_count()
            thread_pool = multiprocessing.pool.ThreadPool(n_workers)

        # important: make a copy of original cipher_vector, so it will not be modified
        cipher_vector = copy.deepcopy(cipher_vector)
        n_threads = thread_pool._processes
        d = self.d
        n = self.n
        quotient = len(cipher_vector) // n_threads
        remainder = len(cipher_vector) % n_threads
        threads = []
        for idx in range(n_threads):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_threads - 1:
                end += remainder
            # Do not place the target function outside the python class, which may
            # cause the decrease of computing performance.
            # This strange phenomenon may due to the reloading of privaty key
            # in target function and delivering self object to target function.
            # Instead, the target function should be within this python class.
            t = threading.Thread(target=RSAPrivateKey._target_dec_vector,
                                 args=(cipher_vector, start, end, d, n))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        return cipher_vector

    @staticmethod
    def _target_dec_vector(ciphertexts, start, end, d, n):
        sub_list = ciphertexts[start:end]
        ciphertexts[start:end] = gmpy2.powmod_list(sub_list, d, n)

    def _raw_decrypt_vector_pool(self, cipher_vector,
                                 using_pool=False, n_workers=None, process_pool=None):
        assert isinstance(cipher_vector, list), \
            "in RSA cryptosystem, cipher_vector can only be a Python list."

        if not using_pool:
            return [self.raw_decrypt(val) for val in cipher_vector]

        if process_pool is None:
            if n_workers is None:
                n_workers = os.cpu_count()
            process_pool = multiprocessing.pool.Pool(n_workers)

        manager = multiprocessing.Manager()
        shared_data = manager.list(cipher_vector)
        d = self.d
        n = self.n
        # the gmpy2.powmod function cannot accept keyword arguments,
        # so you need to write a wrapper function of it.
        # Note that the target function can also not be within this python class
        res = process_pool.map(
            functools.partial(self._target_dec_vector_pool, d=d, n=n),
            shared_data
        )

        return res

    @staticmethod
    def _target_dec_vector_pool(ciphertext, d, n):
        return gmpy2.powmod(ciphertext, d, n)


class PartialRSA(BasePartialCryptoSystem):
    def __init__(self, raw_public_key):
        super(PartialRSA, self).__init__()
        self.pub_key = raw_public_key # for API consistency
        self.pub_key_obj = RSAPublicKey(raw_public_key)
        self.type = Const.RSA

    def encrypt(self, plaintext):
        return self.pub_key_obj.raw_encrypt(plaintext)

    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.pub_key_obj.raw_encrypt_vector(
            plain_vector,
            using_pool,
            n_workers,
            thread_pool
        )

    def inverse(self, plaintext):
        return self.pub_key_obj.raw_inverse(plaintext)

    def mulmod(self, x, y, z):
        return self.pub_key_obj.raw_mulmod(x, y, z)


class RSA(BaseCryptoSystem):
    PRIV_KEY_NAME = 'rsa_priv_key.bin'

    def __init__(self, key_size=1024, e=0x10001, private_key=None):
        super(RSA, self).__init__(key_size)
        raw_public_key, raw_private_key = self._gen_key(key_size, e, private_key)
        # save private key, so it can be loaded back when executing RSA-PSI online protocol
        RSA._save_key(raw_private_key)
        self.pub_key, self.priv_key = raw_public_key, raw_private_key # for API consistency
        self.pub_key_obj = RSAPublicKey(raw_public_key)
        self.priv_key_obj = RSAPrivateKey(raw_private_key)
        self.type = Const.RSA

    def _gen_key(self, key_size, e=0x10001, private_key=None):
        if private_key is None:
            priv_key = CryptoRSA.generate(key_size, e=e)
        else:
            priv_key = private_key
        pub_key = CryptoRSA.construct((priv_key.n, priv_key.e))

        return pub_key, priv_key

    @staticmethod
    def _save_key(key):
        passphrase = 'linkefl'
        encrypted_key = key.export_key(passphrase=passphrase,
                                       pkcs=8,
                                       protection='scryptAndAES128-CBC')
        target_dir = os.path.join(Path.home(), Const.PROJECT_CACHE_DIR)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        full_path = os.path.join(target_dir, RSA.PRIV_KEY_NAME)
        with open(full_path, 'wb') as f:
            f.write(encrypted_key)

    @classmethod
    def from_config(cls, config):
        assert isinstance(config, BaseConfig), 'config object should be an ' \
                                               'instance of BaseConfig class.'
        return cls(key_size=config.KEY_SIZE, e=config.PUB_E)

    @classmethod
    def from_private_key(cls):
        full_path = os.path.join(Path.home(), Const.PROJECT_CACHE_DIR, cls.PRIV_KEY_NAME)

        # check if the private key exists
        if not os.path.exists(full_path):
            raise FileNotFoundError('There is no RSA private key within the'
                                    '~/.linkefl/ directory.')

        # load the private key
        encoded_key = open(full_path, 'rb').read()
        passphrase = 'linkefl'
        private_key = CryptoRSA.import_key(encoded_key, passphrase=passphrase)

        return cls(private_key=private_key)

    def encrypt(self, plaintext):
        return self.pub_key_obj.raw_encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.priv_key_obj.raw_decrypt(ciphertext)

    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.pub_key_obj.raw_encrypt_vector(
            plain_vector,
            using_pool,
            n_workers,
            thread_pool
        )

    def decrypt_vector(self, cipher_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.priv_key_obj.raw_decrypt_vector(
            cipher_vector,
            using_pool,
            n_workers,
            thread_pool
        )

    def sign(self, x):
        return self.decrypt(x)

    def sign_vector(self, X,
                    using_pool=False, n_workers=None, thread_pool=None):
        return self.decrypt_vector(
            X,
            using_pool,
            n_workers,
            thread_pool
        )

    def inverse(self, x):
        return self.pub_key_obj.raw_inverse(x)

    def mulmod(self, x, y, z):
        return self.pub_key_obj.raw_mulmod(x, y, z)


if __name__ == '__main__':
    pass
