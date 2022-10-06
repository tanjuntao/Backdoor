import functools
import math
import multiprocessing.pool
import os
import random
import time
import warnings
from multiprocessing import Manager
from multiprocessing.pool import Pool

import gmpy2
import numpy as np
import torch
from phe import EncodedNumber, EncryptedNumber, paillier
from phe.util import mulmod

from linkefl.config import BaseConfig
from linkefl.crypto.base import CryptoSystem, PartialCryptoSystem


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
    n_squared = public_key.nsquare
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


class PaillierPublicKey:
    def __init__(self, pub_key):
        self.raw_pub_key = pub_key

    @staticmethod
    def _convert_vector(plain_vector):
        if type(plain_vector) == list:
            plain_vector = plain_vector.copy()  # make a copy

        elif type(plain_vector) == np.ndarray:
            dtype = plain_vector.dtype
            if dtype == np.float64:
                plain_vector = [val for val in plain_vector]
            elif dtype in (np.float16, np.float32):
                plain_vector = [float(val) for val in plain_vector]
            elif dtype in (np.int8, np.int16, np.int32, np.int64):
                plain_vector = [int(val) for val in plain_vector]
            else:
                raise TypeError("python-paillier cannot accept numpy array with {}"
                                " dtype".format(dtype))

        elif type(plain_vector) == torch.Tensor:
            dtype = plain_vector.dtype
            if dtype in (torch.float16, torch.float32, torch.float64):
                plain_vector = [val.item() for val in plain_vector]
            elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                plain_vector = [val.item() for val in plain_vector]
            else:
                raise TypeError("python-paillier cannot accept PyTorch Tensor with {} "
                                " dtype".format(dtype))

        else:
            raise TypeError("Only Python list, Numpy Array, and PyTorch Tensor can be"
                            " passed to this method.")

        return plain_vector # always return a Python list

    def raw_encrypt(self, plaintext):
        if type(plaintext) in (np.float16, np.float32):
            plaintext = float(plaintext)
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)

        # note: if a PyTorch Tensor object with only one element needs to be encrypted,
        # e.g., a = torch.Tensor([1]), then a.item() should be passed to this method
        # rather than the tensor itself.
        # Besides, there's no need to convert tensor's item() value to Python float or int.
        return self.raw_pub_key.encrypt(plaintext)

    def raw_fast_encrypt(self, plaintext):
        if not hasattr(self, 'enc_zeros'):
            raise AttributeError("enc_zeros attribute not found, please calculate it first.")
        if type(plaintext) in (np.float16, np.float32):
            plaintext = float(plaintext)
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)

        # TODO: improve security by random choice without replacement
        enc_zero = random.choice(getattr(self, 'enc_zeros'))
        return enc_zero + plaintext

    def raw_encrypt_vector(self, plain_vector,
                           using_pool=False, n_workers=None, thread_pool=None):
        def _encrypt(val):
            # unlike self.raw_encrypt(), there's no need to judge the data type
            return self.raw_pub_key.encrypt(val)

        plain_vector = PaillierPublicKey._convert_vector(plain_vector)

        if not using_pool:
            return [_encrypt(val) for val in plain_vector]

        if thread_pool is None:
            if n_workers is None:
                n_workers = os.cpu_count()
            thread_pool = multiprocessing.pool.ThreadPool(n_workers)

        n_threads = thread_pool._processes
        data_size = len(plain_vector)
        quotient = data_size // n_threads
        remainder = data_size % n_threads
        async_results = []
        for idx in range(n_threads):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_threads - 1:
                end += remainder
            # target function will modify plain_vector in place
            result = thread_pool.apply_async(self._target_enc_vector,
                                             args=(plain_vector, start, end))
            async_results.append(result)
        for idx, result in enumerate(async_results):
            assert result.get() is True, "worker thread did not finish " \
                                         "within default timeout"
        return plain_vector  # is a Python list

    def raw_fast_encrypt_vector(self, plain_vector,
                                using_pool=False, n_workers=None, thread_pool=None):
        def _fast_encrypt(val):
            # unlike self.raw_fast_encrypt(), there's no need to judge the data type
            enc_zero = random.choice(getattr(self, 'enc_zeros'))
            return enc_zero + val

        plain_vector = PaillierPublicKey._convert_vector(plain_vector)

        if not using_pool:
            return [_fast_encrypt(val) for val in plain_vector]
        else:
            raise NotImplementedError('Using multiprocessing.Pool to accelarate fast encryption'
                                      'process is not implemented yet.')

    def _target_enc_vector(self, plaintexts, start, end):
        n = self.raw_pub_key.n
        nsquare = self.raw_pub_key.nsquare
        r_values = [random.SystemRandom().randrange(1, n) for _ in range(end - start)]
        obfuscators = gmpy2.powmod_list(r_values, n, nsquare)

        r_idx = 0
        for k in range(start, end):
            encoding = EncodedNumber.encode(self.raw_pub_key, plaintexts[k])
            nude_ciphertext = (1 + n * encoding.encoding) % nsquare
            ciphertext = mulmod(nude_ciphertext, obfuscators[r_idx], nsquare)
            encrypted_number = EncryptedNumber(self.raw_pub_key, ciphertext, encoding.exponent)
            encrypted_number._EncryptedNumber__is_obfuscated = True
            plaintexts[k] = encrypted_number
            r_idx += 1
        return True

    def raw_encrypt_data(self, plain_data, pool: Pool = None):
        if type(plain_data) == torch.Tensor:
            plain_data = plain_data.numpy()
            data_type = "torch"
            warnings.warn(
                "mixed data type is not supported by pytorch, automatically converting"
                " to numpy"
            )
        elif type(plain_data) == list:
            plain_data = np.array(plain_data)
            data_type = "list"
        elif type(plain_data) == np.ndarray:
            data_type = "numpy"
        else:
            raise TypeError

        shape = plain_data.shape
        flatten_data = plain_data.astype(object).flatten()

        if pool is None or len(flatten_data) < 10000:
            encrypted_data = flatten_data
            assert self._target_enc_data(encrypted_data) is True
            encrypted_data = np.reshape(encrypted_data, shape)
        else:
            print("using pool to speed up")
            n_processes = pool._processes
            manager = Manager()
            shared_data = manager.list(np.array_split(flatten_data, n_processes))

            results = []
            for i in range(n_processes):
                result = pool.apply_async(self._target_enc_data, (shared_data[i],))
                results.append(result)
            for i in range(n_processes):
                assert results[i].get() is True

            def concat(a, b):
                return np.concatenate((a, b))

            encrypted_data = functools.reduce(concat, shared_data)
            encrypted_data = np.reshape(encrypted_data, shape)

        if data_type == "list":
            encrypted_data = encrypted_data.tolist()

        return encrypted_data

    def _target_enc_data(self, shared_vector):
        assert len(shared_vector.shape) == 1

        for i in range(len(shared_vector)):
            shared_vector[i] = self.raw_encrypt(shared_vector[i])
        return True

    def set_enc_zeros(self, enc_zeros):
        setattr(self, 'enc_zeros', enc_zeros)

class PaillierPrivateKey:
    def __init__(self, priv_key):
        self.raw_priv_key = priv_key

    def raw_decrypt(self, ciphertext):
        if isinstance(ciphertext, EncryptedNumber):
            return self.raw_priv_key.decrypt(ciphertext)
        else:
            return ciphertext

    def raw_decrypt_vector(self, cipher_vector,
                           using_pool=False, n_workers=None, thread_pool=None):
        assert type(cipher_vector) in (list, np.ndarray), \
            "cipher_vector's dtype can only be Python list or Numpy array."
        if not using_pool:
            return [self.raw_decrypt(cipher) for cipher in cipher_vector]
        if thread_pool is None:
            if n_workers is None:
                n_workers = os.cpu_count()
            thread_pool = multiprocessing.pool.ThreadPool(n_workers)

        ciphertexts, exponents = [], []
        for enc_number in cipher_vector:
            ciphertexts.append(enc_number.ciphertext(be_secure=False))
            exponents.append(enc_number.exponent)
        n_threads = thread_pool._processes
        data_size = len(cipher_vector)
        quotient = data_size // n_threads
        remainder = data_size % n_threads
        async_results = []
        for idx in range(n_threads):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_workers - 1:
                end += remainder
            # target function will modify ciphertexts in-place
            result = thread_pool.apply_async(self._target_dec_vector,
                                             args=(ciphertexts, exponents, start, end))
            async_results.append(result)
        for idx, result in enumerate(async_results):
            assert result.get() is True, "worker process did not finish " \
                                         "within default timeout"

        return ciphertexts  # always return a Python list

    def _target_dec_vector(self, ciphertexts, exponents, start, end):
        p, psquare, hp = self.raw_priv_key.p, self.raw_priv_key.psquare, self.raw_priv_key.hp
        q, qsquare, hq = self.raw_priv_key.q, self.raw_priv_key.qsquare, self.raw_priv_key.hq
        powmod_p = gmpy2.powmod_list(ciphertexts[start:end], p - 1, psquare) # multi thread
        powmod_q = gmpy2.powmod_list(ciphertexts[start:end], q - 1, qsquare) # multi thread
        public_key = self.raw_priv_key.public_key
        sublist_idx = 0
        for k in range(start, end):
            decrypt_to_p = mulmod(self.raw_priv_key.l_function(powmod_p[sublist_idx], p), hp, p)
            decrypt_to_q = mulmod(self.raw_priv_key.l_function(powmod_q[sublist_idx], q), hq, q)
            encoding = self.raw_priv_key.crt(decrypt_to_p, decrypt_to_q)
            encoding = EncodedNumber(public_key, encoding, exponents[k])
            ciphertexts[k] = encoding.decode() # this will modify ciphertexts(list) in place
            sublist_idx += 1
        return True

    def raw_decrypt_data(self, encrypted_data, pool: Pool = None):
        if type(encrypted_data) == list:
            encrypted_data = np.array(encrypted_data)
            data_type = "list"
        elif type(encrypted_data) == np.ndarray:
            data_type = "numpy"
        else:
            raise TypeError

        shape = encrypted_data.shape
        flatten_data = encrypted_data.astype(object).flatten()

        if pool is None or len(flatten_data) < 10000:
            plain_data = flatten_data
            assert self._target_dec_data(plain_data) is True
            plain_data = np.reshape(plain_data, shape)
        else:
            print("using pool to speed up")
            n_processes = pool._processes
            manager = Manager()
            shared_data = manager.list(np.array_split(flatten_data, n_processes))

            results = []
            for i in range(n_processes):
                result = pool.apply_async(self._target_dec_data, (shared_data[i],))
                results.append(result)
            for result in results:
                assert result.get() is True

            def concat(a, b):
                return np.concatenate((a, b))

            plain_data = functools.reduce(concat, shared_data)
            plain_data = np.reshape(plain_data, shape)
            # plain_data = plain_data.astype(float)

        if data_type == "list":
            plain_data = plain_data.tolist()

        return plain_data

    def _target_dec_data(self, shared_vector):
        assert len(shared_vector.shape) == 1

        for i in range(len(shared_vector)):
            shared_vector[i] = self.raw_decrypt(shared_vector[i])
        return True


class PartialPaillier(PartialCryptoSystem):
    def __init__(self, raw_public_key):
        super(PartialPaillier, self).__init__()
        self.pub_key = PaillierPublicKey(raw_public_key)

    def encrypt(self, plaintext):
        return self.pub_key.raw_encrypt(plaintext)

    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.pub_key.raw_encrypt_vector(
            plain_vector,
            using_pool,
            n_workers,
            thread_pool
        )


class PartialFastPaillier(PartialCryptoSystem):
    def __init__(self, raw_public_key, num_enc_zeros=10000, gen_from_set=True):
        super(PartialFastPaillier, self).__init__()
        self.pub_key = PaillierPublicKey(raw_public_key)
        print("Generating encrypted zeros...")
        enc_zeros = _cal_enc_zeros(raw_public_key, num_enc_zeros, gen_from_set)
        self.pub_key.set_enc_zeros(enc_zeros)
        print("Done!")

    def encrypt(self, plaintext):
        return self.pub_key.raw_fast_encrypt(plaintext)

    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.pub_key.raw_fast_encrypt_vector(
            plain_vector,
            using_pool,
            n_workers,
            thread_pool
        )


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
        raw_public_key, raw_private_key = self._gen_key(key_size)
        self.pub_key = PaillierPublicKey(raw_public_key)
        self.priv_key = PaillierPrivateKey(raw_private_key)

    @classmethod
    def from_config(cls, config):
        assert isinstance(
            config, BaseConfig
        ), "config object should be an instance of BaseConfig class."
        return cls(key_size=config.KEY_SIZE)

    def _gen_key(self, key_size):
        pub_key, priv_key = paillier.generate_paillier_keypair(n_length=key_size)
        return pub_key, priv_key

    def encrypt(self, plaintext):
        return self.pub_key.raw_encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.priv_key.raw_decrypt(ciphertext)

    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.pub_key.raw_encrypt_vector(
            plain_vector,
            using_pool,
            n_workers,
            thread_pool
        )

    def decrypt_vector(self, cipher_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.priv_key.raw_decrypt_vector(
            cipher_vector,
            using_pool,
            n_workers,
            thread_pool
        )

    def encrypt_data(self, plain_data, pool: Pool = None):
        return self.pub_key.raw_encrypt_data(plain_data, pool)

    def decrypt_data(self, encrypted_data, pool: Pool = None):
        return self.priv_key.raw_decrypt_data(encrypted_data, pool)


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
        raw_public_key, raw_private_key = self._gen_key(key_size)
        self.pub_key = PaillierPublicKey(raw_public_key)
        self.priv_key = PaillierPrivateKey(raw_private_key)

        print("Generating encrypted zeros...")
        enc_zeros = _cal_enc_zeros(raw_public_key, num_enc_zeros, gen_from_set)
        self.pub_key.set_enc_zeros(enc_zeros)
        print("Done!")

    @classmethod
    def from_config(cls, config):
        assert isinstance(
            config, BaseConfig
        ), "config object should be an instance of BaseConfig class."
        return cls(
            key_size=config.KEY_SIZE,
            num_enc_zeros=config.NUM_ENC_ZEROS,
            gen_from_set=config.GEN_FROM_SET,
        )

    def _gen_key(self, key_size):
        pub_key, priv_key = paillier.generate_paillier_keypair(n_length=key_size)
        return pub_key, priv_key

    def encrypt(self, plaintext):
        return self.pub_key.raw_fast_encrypt(plaintext)

    def decrypt(self, ciphertext):
        self.priv_key.raw_decrypt(ciphertext)

    def encrypt_vector(self, plain_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.pub_key.raw_fast_encrypt_vector(
            plain_vector,
            using_pool,
            n_workers,
            thread_pool
        )

    def decrypt_vector(self, cipher_vector,
                       using_pool=False, n_workers=None, thread_pool=None):
        return self.priv_key.raw_decrypt_vector(
            cipher_vector,
            using_pool,
            n_workers,
            thread_pool
        )

    def encrypt_data(self, plain_data, pool: Pool = None):
        return self.pub_key.raw_encrypt_data(plain_data, pool)

    def decrypt_data(self, encrypted_data, pool: Pool = None):
        return self.priv_key.raw_decrypt_data(encrypted_data, pool)


if __name__ == "__main__":
    # crypto_system = FastPaillier(num_enc_zeros=64, gen_from_set=True)
    # enc_zero_ = crypto_system._enc_zeros[20]
    # print(type(enc_zero_))
    # print(crypto_system.decrypt(enc_zero_ + 19))

    crypto_system = Paillier(key_size=1024)
    # crypto_system = FastPaillier(num_enc_zeros=128, gen_from_set=True)
    # crypto_system = Plain()
    # item1 = 10
    # item2 = 20.033
    # print(crypto_system.decrypt(crypto_system.encrypt(item1)))
    # print(crypto_system.decrypt(crypto_system.encrypt(item2)))

    # vector = [2.1, 3, 0.0, 0.3]
    # vector = np.arange(2000)
    # vector = np.array([0, 1, 2, 3])
    # vector = list(vector)
    # vector = torch.Tensor([0, 1, 2, 3])
    # vector = torch.arange(10000)
    # vector = np.random.rand(100000)
    # import time
    #
    # print(vector[:10])
    # start = time.time()
    # enc_vector = crypto_system.encrypt_vector(plain_vector=vector, using_mp=True)
    # dec_vector = crypto_system.decrypt_vector(enc_vector, using_mp=True)
    # print(dec_vector[:10])
    # print('Elapsed time: {:.5f}'.format(time.time() - start))

    base_data = np.arange(-5000.0, 5000.0, 0.1).reshape((10, -1))
    # plain_data = base_data.tolist()
    plain_data_ = base_data
    # plain_data = torch.from_numpy(base_data)
    print(plain_data_)

    start_ = time.time()
    pool_ = Pool(6)
    encrypted_data_ = crypto_system.encrypt_data(plain_data_, pool_)
    decrypted_data_ = crypto_system.decrypt_data(encrypted_data_, pool_)
    pool_.close()
    end_ = time.time()
    print(decrypted_data_)
    print(end_ - start_)

    start_ = time.time()
    encrypted_data_ = crypto_system.encrypt_data(plain_data_)
    decrypted_data_ = crypto_system.decrypt_data(encrypted_data_)
    end_ = time.time()
    print(decrypted_data_)
    print(end_ - start_)
