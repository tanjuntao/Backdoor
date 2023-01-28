"""This module implements the Paillier cryptosystem.

The Paillier cryptosystem, invented by and named after Pascal Paillier in 1999, is a
probabilistic asymmetric algorithm for public key cryptography.

The scheme is an additive homomorphic cryptosystem; this means that, give only the
public key and the encrpytion of m1 and m2, one can compute the encryption of m1 + m2.
"""

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
import phe
import torch
from phe import EncodedNumber, EncryptedNumber, paillier
from phe.util import mulmod

from linkefl.base import BaseCryptoSystem, BasePartialCryptoSystem
from linkefl.common.const import Const
from linkefl.config import BaseConfig


def _cal_enc_zeros(public_key, num_enc_zeros, gen_from_set):
    """Calculate pre-computed encrypted zeros for faster encryption later on

    Parameters
    ----------
    public_key
    num_enc_zeros
    gen_from_set

    Returns
    -------

    """

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
    """A wrapper for python-paillier's public key."""

    def __init__(self, raw_pub_key: phe.PaillierPublicKey):
        """Initialize LinkeFL's paillier public key.

        Parameters
        ----------
        raw_pub_key : `phe.PaillierPublicKey`
            `python-paillier`'s public key object.
        """
        self.raw_pub_key = raw_pub_key

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
                raise TypeError(
                    "python-paillier cannot accept numpy array with {} dtype".format(
                        dtype
                    )
                )

        elif type(plain_vector) == torch.Tensor:
            dtype = plain_vector.dtype
            if dtype in (torch.float16, torch.float32, torch.float64):
                plain_vector = [val.item() for val in plain_vector]
            elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                plain_vector = [val.item() for val in plain_vector]
            else:
                raise TypeError(
                    "python-paillier cannot accept PyTorch Tensor with {} "
                    " dtype".format(dtype)
                )

        else:
            raise TypeError(
                "Only Python list, Numpy Array, and PyTorch Tensor can be"
                " passed to this method."
            )

        return plain_vector  # always return a Python list

    def raw_encrypt(self, plaintext):
        """Encrypte single plaintext message via naive Paillier cryptosystem.
        Parameters
        ----------
        plaintext :

        Returns
        -------

        """
        if type(plaintext) in (np.float16, np.float32):
            plaintext = float(plaintext)
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)

        # note: if a PyTorch Tensor object with only one element needs to be encrypted,
        # e.g., a = torch.Tensor([1]), then a.item() should be passed to this method
        # rather than the tensor itself.
        # Besides,
        # there's no need to convert tensor's item() value to Python float or int.
        return self.raw_pub_key.encrypt(plaintext)

    def raw_fast_encrypt(self, plaintext):
        """Encrypt single plaintext message via improved FastPaillier cryptosystem.

        Parameters
        ----------
        plaintext

        Returns
        -------

        """
        if not hasattr(self, "enc_zeros"):
            raise AttributeError(
                "enc_zeros attribute not found, please calculate it first."
            )
        if type(plaintext) in (np.float16, np.float32):
            plaintext = float(plaintext)
        if type(plaintext) in (np.int8, np.int16, np.int32, np.int64):
            plaintext = int(plaintext)

        # TODO: improve security by random choice without replacement
        enc_zero = random.choice(getattr(self, "enc_zeros"))
        return enc_zero + plaintext

    def raw_encrypt_vector(
        self, plain_vector, using_pool=False, n_workers=None, thread_pool=None
    ):
        """Encrypt a vector of plaintext message via naive Paillier cryptosystem.

        Parameters
        ----------
        plain_vector
        using_pool
        n_workers
        thread_pool

        Returns
        -------

        """

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
            result = thread_pool.apply_async(
                self._target_enc_vector, args=(plain_vector, start, end)
            )
            async_results.append(result)
        for idx, result in enumerate(async_results):
            assert (
                result.get() is True
            ), "worker thread did not finish within default timeout"
        return plain_vector  # is a Python list

    def raw_fast_encrypt_vector(
        self, plain_vector, using_pool=False, n_workers=None, process_pool=None
    ):
        """Encrypt a vector of plaintext message via improved FastPaillier cryptosystem.

        Parameters
        ----------
        plain_vector
        using_pool
        n_workers
        process_pool

        Returns
        -------

        """

        def _fast_encrypt(val):
            # unlike self.raw_fast_encrypt(), there's no need to judge the data type
            enc_zero = random.choice(getattr(self, "enc_zeros"))
            return enc_zero + val

        plain_vector = PaillierPublicKey._convert_vector(plain_vector)

        # sequentially
        if not using_pool:
            return [_fast_encrypt(val) for val in plain_vector]

        # sequentially
        using_pool_thresh = (
            10000  # based on empirical evaluations on different machines
        )
        if len(plain_vector) < using_pool_thresh:
            return [_fast_encrypt(val) for val in plain_vector]

        # parallelly
        if process_pool is None:
            if n_workers is None:
                n_workers = os.cpu_count()
            process_pool = multiprocessing.pool.Pool(n_workers)
        n_processes = process_pool._processes
        manager = Manager()
        # important: convert it to object dtype
        plain_vector = np.array(plain_vector).astype(object)
        shared_data = manager.list(
            list(map(manager.list, np.array_split(plain_vector, n_processes)))
        )
        async_results = []
        for i in range(n_processes):
            # this will modify shared_data in place
            result = process_pool.apply_async(
                self._target_fast_enc_vector,
                # all parameters in args will be pickled and passed to child processes,
                # which may be time-consuming
                args=(shared_data[i],),
            )
            async_results.append(result)
        for i in range(n_processes):
            assert (
                async_results[i].get() is True
            ), "worker thread did not finish within default timeout"
        # concat the split sub-arrays into one array and then convert it to Python list
        encrypted_data = functools.reduce(
            lambda a, b: np.concatenate((a, b)), shared_data
        ).tolist()

        return encrypted_data  # always return a Python list

    def _target_enc_vector(self, plaintexts, start, end):
        n = self.raw_pub_key.n
        nsquare = self.raw_pub_key.nsquare
        r_values = [random.SystemRandom().randrange(1, n) for _ in range(end - start)]
        obfuscators = gmpy2.powmod_base_list(r_values, n, nsquare)

        r_idx = 0
        for k in range(start, end):
            encoding = EncodedNumber.encode(self.raw_pub_key, plaintexts[k])
            nude_ciphertext = (1 + n * encoding.encoding) % nsquare
            ciphertext = mulmod(nude_ciphertext, obfuscators[r_idx], nsquare)
            encrypted_number = EncryptedNumber(
                self.raw_pub_key, ciphertext, encoding.exponent
            )
            encrypted_number._EncryptedNumber__is_obfuscated = True
            plaintexts[k] = encrypted_number
            r_idx += 1
        return True

    def _target_fast_enc_vector(self, plaintexts):
        def _fast_encrypt(val):
            # unlike self.raw_fast_encrypt(), there's no need to judge the data type
            enc_zero = random.choice(getattr(self, "enc_zeros"))
            return enc_zero + val

        for idx in range(len(plaintexts)):
            plaintexts[idx] = _fast_encrypt(plaintexts[idx])
        return True

    def raw_encrypt_data(self, plain_data, pool: Pool = None):
        target_func = self._target_enc_data
        return PaillierPublicKey._base_encrypt_data(plain_data, pool, target_func)

    def raw_fast_encrypt_data(self, plain_data, pool: Pool = None):
        target_func = self._target_fast_enc_data
        return PaillierPublicKey._base_encrypt_data(plain_data, pool, target_func)

    @staticmethod
    def _base_encrypt_data(plain_data, pool, target_func):
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
            assert target_func(encrypted_data) is True
            encrypted_data = np.reshape(encrypted_data, shape)
        else:
            print("using pool to speed up")
            n_processes = pool._processes
            manager = Manager()
            shared_data = manager.list(
                list(map(manager.list, np.array_split(flatten_data, n_processes)))
            )

            results = []
            for i in range(n_processes):
                result = pool.apply_async(target_func, (shared_data[i],))
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
        def _encrypt(val):
            # unlike self.raw_encrypt(), there's no need to judge the data type
            return self.raw_pub_key.encrypt(val)

        for i in range(len(shared_vector)):
            shared_vector[i] = _encrypt(shared_vector[i])
        return True

    def _target_fast_enc_data(self, shared_vector):
        def _fast_encrypt(val):
            # unlike self.raw_fast_encrypt(), there's no need to judge the data type
            enc_zero = random.choice(getattr(self, "enc_zeros"))
            return enc_zero + val

        for i in range(len(shared_vector)):
            shared_vector[i] = _fast_encrypt(shared_vector[i])
        return True

    def set_enc_zeros(self, enc_zeros):
        """Set pre-computed encrypted zeros for later usage
        by FastPaillier cryptosystem.

        Parameters
        ----------
        enc_zeros

        Returns
        -------

        """
        setattr(self, "enc_zeros", enc_zeros)


class PaillierPrivateKey:
    """A wrapper for python-paillier's private key.

    Attributes
    ----------
        raw_priv_key: phe.PaillierPrivateKey
            python-paillier's private key object.
    """

    def __init__(self, raw_priv_key: phe.PaillierPrivateKey):
        """Construct LinkeFL's paillier private key.

        Parameters
        ----------
        raw_priv_key
        """
        self.raw_priv_key = raw_priv_key

    def raw_decrypt(self, ciphertext):
        """Decrypt single ciphertext.

        Parameters
        ----------
        ciphertext

        Returns
        -------

        """
        if isinstance(ciphertext, EncryptedNumber):
            return self.raw_priv_key.decrypt(ciphertext)
        else:
            return ciphertext

    def raw_decrypt_vector(
        self, cipher_vector, using_pool=False, n_workers=None, thread_pool=None
    ):
        """Decrypt a vector of ciphertext.

        Parameters
        ----------
        cipher_vector
        using_pool
        n_workers
        thread_pool

        Returns
        -------

        """
        assert type(cipher_vector) in (
            list,
            np.ndarray,
        ), "cipher_vector's dtype can only be Python list or Numpy array."
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
            if idx == n_threads - 1:
                end += remainder
            # target function will modify ciphertexts in-place
            result = thread_pool.apply_async(
                self._target_dec_vector, args=(ciphertexts, exponents, start, end)
            )
            async_results.append(result)
        for idx, result in enumerate(async_results):
            assert (
                result.get() is True
            ), "worker process did not finish within default timeout"

        return ciphertexts  # always return a Python list

    def _target_dec_vector(self, ciphertexts, exponents, start, end):
        p, psquare, hp = (
            self.raw_priv_key.p,
            self.raw_priv_key.psquare,
            self.raw_priv_key.hp,
        )
        q, qsquare, hq = (
            self.raw_priv_key.q,
            self.raw_priv_key.qsquare,
            self.raw_priv_key.hq,
        )
        powmod_p = gmpy2.powmod_base_list(
            ciphertexts[start:end], p - 1, psquare
        )  # multi thread
        powmod_q = gmpy2.powmod_base_list(
            ciphertexts[start:end], q - 1, qsquare
        )  # multi thread
        public_key = self.raw_priv_key.public_key
        sublist_idx = 0
        for k in range(start, end):
            decrypt_to_p = mulmod(
                self.raw_priv_key.l_function(powmod_p[sublist_idx], p), hp, p
            )
            decrypt_to_q = mulmod(
                self.raw_priv_key.l_function(powmod_q[sublist_idx], q), hq, q
            )
            encoding = self.raw_priv_key.crt(decrypt_to_p, decrypt_to_q)
            encoding = EncodedNumber(public_key, encoding, exponents[k])
            ciphertexts[
                k
            ] = encoding.decode()  # this will modify ciphertexts(list) in place
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
            shared_data = manager.list(
                list(map(manager.list, np.array_split(flatten_data, n_processes)))
            )

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
        for i in range(len(shared_vector)):
            shared_vector[i] = self.raw_decrypt(shared_vector[i])
        return True


class PartialPaillier(BasePartialCryptoSystem):
    def __init__(self, raw_public_key):
        super(PartialPaillier, self).__init__()
        self.pub_key = raw_public_key  # for API consistency
        self.pub_key_obj = PaillierPublicKey(raw_public_key)
        self.type = Const.PAILLIER

    def encrypt(self, plaintext):
        return self.pub_key_obj.raw_encrypt(plaintext)

    def encrypt_vector(
        self, plain_vector, using_pool=False, n_workers=None, thread_pool=None
    ):
        return self.pub_key_obj.raw_encrypt_vector(
            plain_vector, using_pool, n_workers, thread_pool
        )


class PartialFastPaillier(BasePartialCryptoSystem):
    def __init__(self, raw_public_key, num_enc_zeros=10000, gen_from_set=True):
        super(PartialFastPaillier, self).__init__()
        self.pub_key = raw_public_key
        self.pub_key_obj = PaillierPublicKey(raw_public_key)
        self.type = Const.FAST_PAILLIER

        print("Generating encrypted zeros...")
        enc_zeros = _cal_enc_zeros(raw_public_key, num_enc_zeros, gen_from_set)
        self.pub_key_obj.set_enc_zeros(enc_zeros)
        print("Done!")

    def encrypt(self, plaintext):
        return self.pub_key_obj.raw_fast_encrypt(plaintext)

    def encrypt_vector(
        self, plain_vector, using_pool=False, n_workers=None, process_pool=None
    ):
        return self.pub_key_obj.raw_fast_encrypt_vector(
            plain_vector, using_pool, n_workers, process_pool
        )


class Paillier(BaseCryptoSystem):
    """Paillier additive homomorphic cryptosystem.

    Paillier cryptosystem satisfies additive homomorphism,
    which means that:

    .. math::
        E(u) + E(v) = E(u+v)  and  D(E(u+v)) = u + v

    .. math::
        E(u) * v = E(u * v)  and  D(E(u*v)) = u * v

    See Also
    --------
    PartialPaillier : Paillier cryptosystem for passive party.
    FastPaillier : A faster paillier cryptosystem constructed
        by pre-computed encrypted zeros.

    Notes
    ------
    ``+`` and ``*`` in the above equations are not arithmetic symbols,
    but notations to represent addition and multiplication operations on
    encrypted data.

    References
    ----------
    The Paillier cryptosystem is invented and named after Pascal Paillier in 1999,
    the original paper is present in [1]_.

    .. [1] Paillier, P. (1999, May). Public-key cryptosystems based on composite
        degree residuosity classes. In International conference on the theory and
        applications of cryptographic techniques (pp. 223-238).
        Springer, Berlin, Heidelberg.

    Examples
    --------
    >>> from linkefl.crypto import Paillier
    >>> paillier_crypto = Paillier(key_size=1024)
    >>> a = 10.1
    >>> enc_a = paillier_crypto.encrypt(a)
    >>> enc_a
    <phe.paillier.EncryptedNumber object at 0x112f2bca0>
    >>> dec_a = paillier_crypto.decrypt(enc_a)
    >>> dec_a
    10.1

    """

    def __init__(self, key_size=1024):
        super(Paillier, self).__init__(key_size)
        raw_public_key, raw_private_key = self._gen_key(key_size)
        self.pub_key, self.priv_key = (
            raw_public_key,
            raw_private_key,
        )  # for API consistency
        self.pub_key_obj = PaillierPublicKey(raw_public_key)
        self.priv_key_obj = PaillierPrivateKey(raw_private_key)
        self.type = Const.PAILLIER

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
        return self.pub_key_obj.raw_encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.priv_key_obj.raw_decrypt(ciphertext)

    def encrypt_vector(
        self, plain_vector, using_pool=False, n_workers=None, thread_pool=None
    ):
        return self.pub_key_obj.raw_encrypt_vector(
            plain_vector, using_pool, n_workers, thread_pool
        )

    def decrypt_vector(
        self, cipher_vector, using_pool=False, n_workers=None, thread_pool=None
    ):
        return self.priv_key_obj.raw_decrypt_vector(
            cipher_vector, using_pool, n_workers, thread_pool
        )

    def encrypt_data(self, plain_data, pool: Pool = None):
        return self.pub_key_obj.raw_encrypt_data(plain_data, pool)

    def decrypt_data(self, encrypted_data, pool: Pool = None):
        return self.priv_key_obj.raw_decrypt_data(encrypted_data, pool)


class FastPaillier(BaseCryptoSystem):
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
        self.pub_key, self.priv_key = raw_public_key, raw_private_key
        self.pub_key_obj = PaillierPublicKey(raw_public_key)
        self.priv_key_obj = PaillierPrivateKey(raw_private_key)
        self.type = Const.FAST_PAILLIER

        print("Generating encrypted zeros...")
        enc_zeros = _cal_enc_zeros(raw_public_key, num_enc_zeros, gen_from_set)
        self.pub_key_obj.set_enc_zeros(enc_zeros)
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
        return self.pub_key_obj.raw_fast_encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.priv_key_obj.raw_decrypt(ciphertext)

    def encrypt_vector(
        self, plain_vector, using_pool=False, n_workers=None, process_pool=None
    ):
        return self.pub_key_obj.raw_fast_encrypt_vector(
            plain_vector, using_pool, n_workers, process_pool
        )

    def decrypt_vector(
        self, cipher_vector, using_pool=False, n_workers=None, thread_pool=None
    ):
        return self.priv_key_obj.raw_decrypt_vector(
            cipher_vector, using_pool, n_workers, thread_pool
        )

    def encrypt_data(self, plain_data, pool: Pool = None):
        return self.pub_key_obj.raw_fast_encrypt_data(plain_data, pool)

    def decrypt_data(self, encrypted_data, pool: Pool = None):
        return self.priv_key_obj.raw_decrypt_data(encrypted_data, pool)


def cipher_matmul(
    cipher_matrix: np.ndarray,
    plain_matrix: np.ndarray,
    executor_pool: multiprocessing.pool.ThreadPool,
    scheduler_pool: multiprocessing.pool.ThreadPool,
):
    """

    Parameters
    ----------
    cipher_matrix
    plain_matrix
    executor_pool
    scheduler_pool

    Returns
    -------

    """
    assert cipher_matrix.shape[1] == plain_matrix.shape[0], (
        "Matrix shape dismatch error. cipher_matrix shape is {}, plain_matrix shape"
        " is {}".format(cipher_matrix.shape, plain_matrix.shape)
    )

    result_matrix = []
    for i in range(len(cipher_matrix)):
        curr_result = _cipher_mat_vec_product(
            cipher_vector=cipher_matrix[i],
            plain_matrix=plain_matrix,
            executor_pool=executor_pool,
            scheduler_pool=scheduler_pool,
        )
        result_matrix.append(curr_result)

    return np.array(result_matrix)


def _cipher_mat_vec_product(cipher_vector, plain_matrix, executor_pool, scheduler_pool):
    height, width = plain_matrix.shape

    # 1. multiply each raw of plain_matrix with its corresponding enc value
    enc_result = [None] * height
    data_size = height
    n_schedulers = scheduler_pool._processes
    quotient = data_size // n_schedulers
    remainder = data_size % n_schedulers
    async_results = []
    for idx in range(n_schedulers):
        start = idx * quotient
        end = (idx + 1) * quotient
        if idx == n_schedulers - 1:
            end += remainder
        # this will modify enc_result in place
        result = scheduler_pool.apply_async(
            _target_row_mul,
            args=(cipher_vector, plain_matrix, enc_result, start, end, executor_pool),
        )
        async_results.append(result)
    for result in async_results:
        assert result.get() is True

    # 2. transpose enc_result
    enc_result = np.array(enc_result).transpose()

    # 3. average enc_result
    avg_result = [None] * width
    data_size = width
    n_schedulers = scheduler_pool._processes
    quotient = data_size // n_schedulers
    remainder = data_size % n_schedulers
    async_results = []
    for idx in range(n_schedulers):
        start = idx * quotient
        end = (idx + 1) * quotient
        if idx == n_schedulers - 1:
            end += remainder
        # this will modify avg_result in place
        result = scheduler_pool.apply_async(
            _target_row_add, args=(enc_result, avg_result, start, end, executor_pool)
        )
        async_results.append(result)
    for result in async_results:
        assert result.get() is True

    return np.array(avg_result)


def _target_row_mul(enc_vector, plain_matrix, enc_result, start, end, executor_pool):
    for k in range(start, end):
        enc_row = fast_mul_ciphers(plain_matrix[k], enc_vector[k], executor_pool)
        enc_result[k] = enc_row
    return True


def _target_row_add(enc_result, avg_result, start, end, executor_pool):
    for k in range(start, end):
        row_sum = fast_add_ciphers(enc_result[k], executor_pool)
        avg_result[k] = row_sum
    return True


def fast_add_ciphers(cipher_vector, thread_pool=None):
    """

    Parameters
    ----------
    cipher_vector
    thread_pool

    Returns
    -------

    """
    assert type(cipher_vector) in (
        list,
        np.ndarray,
    ), "cipher_vector's dtype can only be Python list or Numpy array."
    exp2cipher = {}
    for enc_number in cipher_vector:
        ciphertext = enc_number.ciphertext(be_secure=False)
        exponent = enc_number.exponent
        if exponent not in exp2cipher:
            exp2cipher[exponent] = [gmpy2.mpz(ciphertext)]
        else:
            exp2cipher[exponent].append(gmpy2.mpz(ciphertext))

    if thread_pool is None:
        n_workers = min(os.cpu_count(), len(exp2cipher))
        thread_pool = multiprocessing.pool.ThreadPool(n_workers)

    min_exp = min(exp2cipher.keys())
    # print(f"length: {len(exp2cipher)}, "
    #       f"min: {min(exp2cipher.keys())}, "
    #       f"max: {max(exp2cipher.keys())}, "
    #       f"base: {EncodedNumber.BASE}")
    base = EncodedNumber.BASE
    public_key = cipher_vector[0].public_key
    nsquare = public_key.nsquare
    async_results = []
    for exp, ciphers in exp2cipher.items():
        result = thread_pool.apply_async(
            _target_add_ciphers, args=(ciphers, exp, min_exp, base, nsquare)
        )
        async_results.append(result)
    final_ciphertext = gmpy2.mpz(1)
    for result in async_results:
        final_ciphertext = gmpy2.mod(gmpy2.mul(result.get(), final_ciphertext), nsquare)

    return EncryptedNumber(public_key, int(final_ciphertext), min_exp)


def _target_add_ciphers(ciphertexts, curr_exp, min_exp, base, nsquare):
    multiplier = pow(base, curr_exp - min_exp)
    aligned_ciphers = gmpy2.powmod_base_list(ciphertexts, multiplier, nsquare)
    result = gmpy2.mpz(1)
    for ciphertext in aligned_ciphers:
        result = gmpy2.mod(gmpy2.mul(result, ciphertext), nsquare)
    return result


def fast_mul_ciphers(plain_vector, cipher, thread_pool=None):
    """

    Parameters
    ----------
    plain_vector
    cipher
    thread_pool

    Returns
    -------

    """
    assert type(plain_vector) in (
        list,
        np.ndarray,
    ), "plain_vector's dtype can only be Python list or Numpy array."
    if thread_pool is None:
        n_workers = 2
        thread_pool = multiprocessing.pool.ThreadPool(n_workers)
    public_key = cipher.public_key
    n, nsquare, max_int = public_key.n, public_key.nsquare, public_key.max_int
    ciphertext, exponent = cipher.ciphertext(be_secure=False), cipher.exponent
    ciphertext_inverse = gmpy2.invert(ciphertext, nsquare)

    pos_idxs, neg_idxs = [], []
    pos_exps, neg_exps = [], []
    for i, encoded_number in enumerate(plain_vector):
        encoding = encoded_number.encoding
        if n - max_int <= encoding:
            neg_idxs.append(i)
            neg_exps.append(n - encoding)
        else:
            pos_idxs.append(i)
            pos_exps.append(encoding)

    async_results = []
    for mode in ("pos", "neg"):
        if mode == "pos":
            result = thread_pool.apply_async(
                _target_mul_ciphers, args=(ciphertext, pos_exps, nsquare)
            )
        else:
            result = thread_pool.apply_async(
                _target_mul_ciphers, args=(ciphertext_inverse, neg_exps, nsquare)
            )
        async_results.append(result)
    pos_ciphertexts = async_results[0].get()
    neg_ciphertexts = async_results[1].get()

    result_vector = [None] * len(plain_vector)  # plain_vector is never modified
    for pos_i, pos_cipher in zip(pos_idxs, pos_ciphertexts):
        exp = plain_vector[pos_i].exponent + exponent
        result_vector[pos_i] = EncryptedNumber(public_key, int(pos_cipher), exp)
    for neg_i, neg_cipher in zip(neg_idxs, neg_ciphertexts):
        exp = plain_vector[neg_i].exponent + exponent
        result_vector[neg_i] = EncryptedNumber(public_key, int(neg_cipher), exp)

    return result_vector


def _target_mul_ciphers(base, exps, nsquare):
    return gmpy2.powmod_exp_list(base, exps, nsquare)


def encode(
    raw_data: np.ndarray, raw_pub_key: phe.PaillierPublicKey, precision: float = 0.001
):
    """

    Parameters
    ----------
    raw_data
    raw_pub_key
    precision

    Returns
    -------

    """
    data_flat = raw_data.flatten()
    # remember to use val.item(), otherwise,
    # "TypeError('argument should be a string or a Rational instance'" will be raised
    data_encode = [
        EncodedNumber.encode(raw_pub_key, val.item(), precision=precision)
        for val in data_flat
    ]
    return np.array(data_encode).reshape(raw_data.shape)


if __name__ == "__main__":
    # crypto_system = FastPaillier(num_enc_zeros=64, gen_from_set=True)
    # enc_zero_ = crypto_system._enc_zeros[20]
    # print(type(enc_zero_))
    # print(crypto_system.decrypt(enc_zero_ + 19))

    _crypto_system = Paillier(key_size=1024)
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
    encrypted_data_ = _crypto_system.encrypt_data(plain_data_, pool_)
    decrypted_data_ = _crypto_system.decrypt_data(encrypted_data_, pool_)
    pool_.close()
    end_ = time.time()
    print(decrypted_data_)
    print(end_ - start_)

    start_ = time.time()
    encrypted_data_ = _crypto_system.encrypt_data(plain_data_)
    decrypted_data_ = _crypto_system.decrypt_data(encrypted_data_)
    end_ = time.time()
    print(decrypted_data_)
    print(end_ - start_)
