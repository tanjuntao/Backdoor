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
from multiprocessing.pool import Pool, ThreadPool
from typing import Dict, List, Optional, Tuple, Union

import gmpy2
import numpy as np
import phe
import torch
from phe import EncodedNumber, EncryptedNumber, paillier
from phe.util import mulmod

from linkefl.base import BaseCryptoSystem, BasePartialCryptoSystem
from linkefl.common.const import Const

NumpyTypes = Union[np.float16, np.float32, np.int8, np.int16, np.int32, np.int64]


def _gen_enc_zeros(public_key, num_enc_zeros, gen_from_set):
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

    def raw_encrypt_vector(self, plain_vector, thread_pool=None, num_workers=1):
        """Encrypt a vector of plaintext message via naive Paillier cryptosystem.

        Parameters
        ----------
        plain_vector
        thread_pool
        num_workers

        Returns
        -------

        """

        def _encrypt(val):
            # unlike self.raw_encrypt(), there's no need to judge the data type
            return self.raw_pub_key.encrypt(val)

        assert (
            num_workers >= 1
        ), f"number of workers should >=1, but got {num_workers} instead."
        plain_vector = PaillierPublicKey._convert_vector(plain_vector)

        if thread_pool is None and num_workers == 1:
            return [_encrypt(val) for val in plain_vector]

        create_pool = False
        if thread_pool is None:
            create_pool = True
            thread_pool = multiprocessing.pool.ThreadPool(num_workers)

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

        if create_pool:
            thread_pool.close()
        return plain_vector  # is a Python list

    def raw_fast_encrypt_vector(self, plain_vector, process_pool=None, num_workers=1):
        """Encrypt a vector of plaintext message via improved FastPaillier cryptosystem.

        Parameters
        ----------
        plain_vector
        process_pool
        num_workers

        Returns
        -------

        """

        def _fast_encrypt(val):
            # unlike self.raw_fast_encrypt(), there's no need to judge the data type
            enc_zero = random.choice(getattr(self, "enc_zeros"))
            return enc_zero + val

        assert (
            num_workers >= 1
        ), f"number of workers should >=1, but got {num_workers} instead."
        plain_vector = PaillierPublicKey._convert_vector(plain_vector)

        # sequentially
        if process_pool is None and num_workers == 1:
            return [_fast_encrypt(val) for val in plain_vector]

        # sequentially
        using_pool_thresh = (
            10000  # based on empirical evaluations on different machines
        )
        if len(plain_vector) < using_pool_thresh:
            warnings.warn(
                "It's not recommended to use multiprocessing when the length "
                "of plain_vector is less than 10000. Still use single process.",
                stacklevel=3,
            )
            return [_fast_encrypt(val) for val in plain_vector]

        # parallelly
        create_pool = False
        if process_pool is None:
            create_pool = True
            process_pool = multiprocessing.pool.Pool(num_workers)
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

        if create_pool:
            process_pool.close()
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

    def raw_encrypt_data(self, plain_data, process_pool=None):
        target_func = self._target_enc_data
        return PaillierPublicKey._base_encrypt_data(
            plain_data, process_pool, target_func
        )

    def raw_fast_encrypt_data(self, plain_data, process_pool=None):
        target_func = self._target_fast_enc_data
        return PaillierPublicKey._base_encrypt_data(
            plain_data, process_pool, target_func
        )

    @staticmethod
    def _base_encrypt_data(plain_data, process_pool, target_func):
        if type(plain_data) == torch.Tensor:
            plain_data = plain_data.numpy()
            data_type = "torch"
            warnings.warn(
                "mixed data type is not supported by pytorch, automatically converting"
                " to numpy",
                stacklevel=4,
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

        if process_pool is None or len(flatten_data) < 10000:
            encrypted_data = flatten_data
            assert target_func(encrypted_data) is True
            encrypted_data = np.reshape(encrypted_data, shape)
        else:
            print("using pool to speed up")
            n_processes = process_pool._processes
            manager = Manager()
            shared_data = manager.list(
                list(map(manager.list, np.array_split(flatten_data, n_processes)))
            )

            results = []
            for i in range(n_processes):
                result = process_pool.apply_async(target_func, (shared_data[i],))
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

    def raw_decrypt_vector(self, cipher_vector, thread_pool=None, num_workers=1):
        """Decrypt a vector of ciphertext.

        Parameters
        ----------
        cipher_vector
        thread_pool
        num_workers

        Returns
        -------

        """
        assert type(cipher_vector) in (
            list,
            np.ndarray,
        ), (
            "cipher_vector's dtype can only be Python list or Numpy array, but got"
            f" {type(cipher_vector)} instead."
        )
        assert (
            num_workers >= 1
        ), f"number of workers should >=1, but got {num_workers} instead."

        if thread_pool is None and num_workers == 1:
            return [self.raw_decrypt(cipher) for cipher in cipher_vector]

        create_pool = False
        if thread_pool is None:
            create_pool = True
            thread_pool = multiprocessing.pool.ThreadPool(num_workers)

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

        if create_pool:
            thread_pool.close()
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

    def raw_decrypt_data(self, encrypted_data, process_pool=None):
        if type(encrypted_data) == list:
            encrypted_data = np.array(encrypted_data)
            data_type = "list"
        elif type(encrypted_data) == np.ndarray:
            data_type = "numpy"
        else:
            raise TypeError

        shape = encrypted_data.shape
        flatten_data = encrypted_data.astype(object).flatten()

        if process_pool is None or len(flatten_data) < 10000:
            plain_data = flatten_data
            assert self._target_dec_data(plain_data) is True
            plain_data = np.reshape(plain_data, shape)
        else:
            print("using pool to speed up")
            n_processes = process_pool._processes
            manager = Manager()
            shared_data = manager.list(
                list(map(manager.list, np.array_split(flatten_data, n_processes)))
            )

            results = []
            for i in range(n_processes):
                result = process_pool.apply_async(
                    self._target_dec_data, (shared_data[i],)
                )
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
    def __init__(self, raw_public_key: phe.PaillierPublicKey):
        super(PartialPaillier, self).__init__()
        self.pub_key: phe.PaillierPublicKey = raw_public_key  # for API consistency
        self.pub_key_obj = PaillierPublicKey(raw_public_key)
        self.type: str = Const.PAILLIER

    def encrypt(self, plaintext: Union[NumpyTypes, float, int]) -> EncryptedNumber:
        return self.pub_key_obj.raw_encrypt(plaintext)

    def encrypt_vector(
        self,
        plain_vector: Union[List[Union[float, int]], np.ndarray, torch.Tensor],
        *,
        pool: Optional[ThreadPool] = None,
        num_workers: int = 1,
    ) -> List[EncryptedNumber]:
        return self.pub_key_obj.raw_encrypt_vector(plain_vector, pool, num_workers)


class PartialFastPaillier(BasePartialCryptoSystem):
    def __init__(
        self,
        raw_public_key: phe.PaillierPublicKey,
        num_enc_zeros: int = 10000,
        gen_from_set: bool = True,
    ):
        super(PartialFastPaillier, self).__init__()
        self.pub_key: phe.PaillierPublicKey = raw_public_key
        self.pub_key_obj = PaillierPublicKey(raw_public_key)
        self.type: str = Const.FAST_PAILLIER

        print("Generating encrypted zeros...")
        enc_zeros = _gen_enc_zeros(raw_public_key, num_enc_zeros, gen_from_set)
        self.pub_key_obj.set_enc_zeros(enc_zeros)
        print("Done!")

    def encrypt(self, plaintext: Union[NumpyTypes, float, int]) -> EncryptedNumber:
        return self.pub_key_obj.raw_fast_encrypt(plaintext)

    def encrypt_vector(
        self,
        plain_vector: Union[List[Union[float, int]], np.ndarray, torch.Tensor],
        *,
        pool: Optional[Pool] = None,
        num_workers: int = 1,
    ) -> List[EncryptedNumber]:
        return self.pub_key_obj.raw_fast_encrypt_vector(plain_vector, pool, num_workers)


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

    def __init__(self, key_size: int = 1024):
        super(Paillier, self).__init__(key_size)
        raw_public_key, raw_private_key = self._gen_key(key_size)
        self.pub_key: phe.PaillierPublicKey = raw_public_key
        self.priv_key: phe.PaillierPrivateKey = raw_private_key
        self.pub_key_obj = PaillierPublicKey(raw_public_key)
        self.priv_key_obj = PaillierPrivateKey(raw_private_key)
        self.type: str = Const.PAILLIER

    def _gen_key(self, key_size):
        pub_key, priv_key = paillier.generate_paillier_keypair(n_length=key_size)
        return pub_key, priv_key

    def encrypt(self, plaintext: Union[NumpyTypes, float, int]) -> EncryptedNumber:
        return self.pub_key_obj.raw_encrypt(plaintext)

    def decrypt(
        self, ciphertext: Union[EncryptedNumber, float, int]
    ) -> Union[float, int]:
        return self.priv_key_obj.raw_decrypt(ciphertext)

    def encrypt_vector(
        self,
        plain_vector: Union[List[Union[float, int]], np.ndarray, torch.Tensor],
        *,
        pool: Optional[ThreadPool] = None,
        num_workers: int = 1,
    ) -> List[EncryptedNumber]:
        return self.pub_key_obj.raw_encrypt_vector(plain_vector, pool, num_workers)

    def decrypt_vector(
        self,
        cipher_vector: Union[List[EncryptedNumber], np.ndarray],
        *,
        pool: Optional[ThreadPool] = None,
        num_workers: int = 1,
    ) -> List[Union[float, int]]:
        return self.priv_key_obj.raw_decrypt_vector(
            cipher_vector,
            pool,
            num_workers,
        )

    def encrypt_data(
        self,
        plain_data: Union[List[Union[float, int]], np.ndarray, torch.Tensor],
        *,
        pool: Optional[Pool] = None,
    ) -> Union[List[EncryptedNumber], np.ndarray]:
        return self.pub_key_obj.raw_encrypt_data(plain_data, pool)

    def decrypt_data(
        self,
        encrypted_data: Union[List[Union[EncryptedNumber, float, int]], np.ndarray],
        *,
        pool: Optional[Pool] = None,
    ) -> Union[List[Union[float, int]], np.ndarray]:
        return self.priv_key_obj.raw_decrypt_data(encrypted_data, pool)


class FastPaillier(BaseCryptoSystem):
    """
    Faster paillier encryption using pre-computed encrypted zeros.
    """

    def __init__(
        self,
        key_size: int = 1024,
        num_enc_zeros: int = 10000,
        gen_from_set: bool = True,
    ):
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
        self.pub_key: phe.PaillierPublicKey = raw_public_key
        self.priv_key: phe.PaillierPrivateKey = raw_private_key
        self.pub_key_obj = PaillierPublicKey(raw_public_key)
        self.priv_key_obj = PaillierPrivateKey(raw_private_key)
        self.type: str = Const.FAST_PAILLIER

        print("Generating encrypted zeros...")
        enc_zeros = _gen_enc_zeros(raw_public_key, num_enc_zeros, gen_from_set)
        self.pub_key_obj.set_enc_zeros(enc_zeros)
        print("Done!")

    def _gen_key(self, key_size):
        pub_key, priv_key = paillier.generate_paillier_keypair(n_length=key_size)
        return pub_key, priv_key

    def encrypt(self, plaintext: Union[NumpyTypes, float, int]) -> EncryptedNumber:
        return self.pub_key_obj.raw_fast_encrypt(plaintext)

    def decrypt(
        self,
        ciphertext: Union[EncryptedNumber, float, int],
    ) -> Union[float, int]:
        return self.priv_key_obj.raw_decrypt(ciphertext)

    def encrypt_vector(
        self,
        plain_vector: Union[List[Union[float, int]], np.ndarray, torch.Tensor],
        *,
        pool: Optional[Pool] = None,
        num_workers: int = 1,
    ) -> List[EncryptedNumber]:
        return self.pub_key_obj.raw_fast_encrypt_vector(plain_vector, pool, num_workers)

    def decrypt_vector(
        self,
        cipher_vector: Union[List[EncryptedNumber], np.ndarray],
        *,
        pool: Optional[ThreadPool] = None,
        num_workers: int = 1,
    ) -> List[Union[float, int]]:
        return self.priv_key_obj.raw_decrypt_vector(
            cipher_vector,
            pool,
            num_workers,
        )

    def encrypt_data(
        self,
        plain_data: Union[List[Union[float, int]], np.ndarray, torch.Tensor],
        *,
        pool: Optional[Pool] = None,
    ) -> Union[List[EncryptedNumber], np.ndarray]:
        return self.pub_key_obj.raw_fast_encrypt_data(plain_data, pool)

    def decrypt_data(
        self,
        encrypted_data: Union[List[Union[EncryptedNumber, float, int]], np.ndarray],
        *,
        pool: Optional[Pool] = None,
    ) -> Union[List[Union[float, int]], np.ndarray]:
        return self.priv_key_obj.raw_decrypt_data(encrypted_data, pool)


def fast_cipher_sum(
    cipher_vector: Union[List[EncryptedNumber], np.ndarray]
) -> EncryptedNumber:
    """

    Parameters
    ----------
    cipher_vector

    Returns
    -------

    """
    base = EncodedNumber.BASE
    public_key = cipher_vector[0].public_key
    nsquare = public_key.nsquare

    # cluster the ciphers according to their exponent
    exp2cluster = {}
    for enc_val in cipher_vector:
        if enc_val.exponent not in exp2cluster:
            exp2cluster[enc_val.exponent] = [gmpy2.mpz(enc_val.ciphertext(False))]
        else:
            exp2cluster[enc_val.exponent].append(gmpy2.mpz(enc_val.ciphertext(False)))

    # sum the ciphers (no need to align exponent) in each cluster
    exp2summation = {exp: None for exp in exp2cluster.keys()}
    for exp, cluster in exp2cluster.items():
        cluster_sum = gmpy2.mpz(1)
        for cipher in cluster:
            cluster_sum = gmpy2.mod(gmpy2.mul(cluster_sum, cipher), nsquare)
        exp2summation[exp] = cluster_sum

    # align exponent across clusters
    min_exp = min(exp2summation.keys())
    for exp, cipher in exp2summation.items():
        multiplier = pow(base, exp - min_exp)
        new_cipher = gmpy2.powmod(cipher, multiplier, nsquare)
        exp2summation[exp] = new_cipher

    # finally sum the aligned ciphers together
    final_cipher = gmpy2.mpz(1)
    for cipher in exp2summation.values():
        final_cipher = gmpy2.mod(gmpy2.mul(final_cipher, cipher), nsquare)

    return EncryptedNumber(public_key, int(final_cipher), min_exp)


def fast_cipher_matmul(
    cipher_matrix: np.ndarray,
    encode_matrix: np.ndarray,
    encode_mappings: List[Dict[str, List[int]]],
    executor_pool: ThreadPool,
    scheduler_pool: ThreadPool,
) -> np.ndarray:
    """

    Parameters
    ----------
    cipher_matrix
    encode_matrix
    encode_mappings
    executor_pool
    scheduler_pool

    Returns
    -------

    """
    assert cipher_matrix.shape[1] == encode_matrix.shape[0], (
        "Matrix shape dismatch error. cipher_matrix shape is {}, encode_matrix shape"
        " is {}".format(cipher_matrix.shape, encode_matrix.shape)
    )

    if executor_pool is None and scheduler_pool is None:
        return np.matmul(cipher_matrix, encode_matrix)

    result_matrix = []
    for i in range(len(cipher_matrix)):
        curr_result = _cipher_mat_vec_product(
            cipher_vector=cipher_matrix[i],
            encode_matrix=encode_matrix,
            encode_mappings=encode_mappings,
            executor_pool=executor_pool,
            scheduler_pool=scheduler_pool,
        )
        result_matrix.append(curr_result)

    return np.array(result_matrix)


def _cipher_mat_vec_product(
    cipher_vector, encode_matrix, encode_mappings, executor_pool, scheduler_pool
):
    height, width = encode_matrix.shape

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
            args=(
                cipher_vector,
                encode_matrix,
                encode_mappings,
                enc_result,
                start,
                end,
                executor_pool,
            ),
        )
        async_results.append(result)
    for result in async_results:
        assert result.get() is True

    # 2. summation alone the vertical axis
    # Note: when summing each column of enc_result, using a simple for-loop will be
    # faster than using multiprocessing, this may be because the parameters need to be
    # pickled and transferred to child processes, which is time-consuming.
    enc_result = np.array(enc_result)
    avg_result = [fast_cipher_sum(enc_result[:, i]) for i in range(width)]

    return np.array(avg_result)


def _target_row_mul(
    enc_vector, encode_matrix, encode_mappings, enc_result, start, end, executor_pool
):
    for k in range(start, end):
        enc_row = fast_cipher_mul(
            encode_matrix[k], enc_vector[k], encode_mappings[k], executor_pool
        )
        enc_result[k] = enc_row
    return True


def fast_cipher_mul(
    encode_vector: Union[List[EncodedNumber], np.ndarray],
    cipher: EncryptedNumber,
    encode_mapping: Dict[str, List[int]],
    thread_pool: Optional[ThreadPool] = None,
) -> List[Union[EncryptedNumber, None]]:
    """

    Parameters
    ----------
    encode_vector
    cipher
    encode_mapping
    thread_pool

    Returns
    -------

    """
    assert type(encode_vector) in (
        list,
        np.ndarray,
    ), "encode_vector's dtype can only be Python list or Numpy array."
    create_pool = False
    if thread_pool is None:
        create_pool = True
        n_workers = os.cpu_count()
        thread_pool = multiprocessing.pool.ThreadPool(n_workers)
    public_key = cipher.public_key
    nsquare = public_key.nsquare
    ciphertext, exponent = cipher.ciphertext(be_secure=False), cipher.exponent
    ciphertext_inverse = gmpy2.invert(ciphertext, nsquare)

    pos_idxs, neg_idxs = encode_mapping["pos_idxs"], encode_mapping["neg_idxs"]
    pos_exps, neg_exps = encode_mapping["pos_exps"], encode_mapping["neg_exps"]
    pos_result = thread_pool.apply_async(
        _target_mul_ciphers, args=(ciphertext, pos_exps, nsquare)
    )
    neg_result = thread_pool.apply_async(
        _target_mul_ciphers, args=(ciphertext_inverse, neg_exps, nsquare)
    )
    pos_ciphertexts = pos_result.get()
    neg_ciphertexts = neg_result.get()

    result_vector = [None] * len(encode_vector)  # plain_vector is never modified
    for pos_i, pos_cipher in zip(pos_idxs, pos_ciphertexts):
        exp = encode_vector[pos_i].exponent + exponent
        result_vector[pos_i] = EncryptedNumber(public_key, int(pos_cipher), exp)
    for neg_i, neg_cipher in zip(neg_idxs, neg_ciphertexts):
        exp = encode_vector[neg_i].exponent + exponent
        result_vector[neg_i] = EncryptedNumber(public_key, int(neg_cipher), exp)

    if create_pool:
        thread_pool.close()

    return result_vector


def _target_mul_ciphers(base, exps, nsquare):
    return gmpy2.powmod_exp_list(base, exps, nsquare)


def encode(
    raw_data: np.ndarray,
    raw_pub_key: phe.PaillierPublicKey,
    *,
    precision: float = 0.001,
    pool: Optional[Pool] = None,
    num_workers: int = 1,
) -> Tuple[np.ndarray, List[Dict[str, List[int]]]]:
    """

    Parameters
    ----------
    raw_data
    raw_pub_key
    precision
    pool:
    num_workers:

    Returns
    -------

    """
    assert len(raw_data.shape) == 2, (
        "raw_data can only be a numpy array with 2 axis, but got a shape"
        f" {raw_data.shape} instead."
    )
    data_encode = []
    encode_mappings = []
    # remember to use val.item(), otherwise,
    # "TypeError('argument should be a string or a Rational instance'" will be raised
    if raw_data.size < 1000 or (pool is None and num_workers == 1):
        if raw_data.size < 1000 and pool is not None:
            warnings.warn(
                "It's not recommended to use multiprocessing when the number of"
                " elements inraw_data is less than 1000. Still use single process.",
                stacklevel=2,
            )
        n_rows, n_cols = raw_data.shape
        for i in range(n_rows):
            encode_vector, vector_mapping = _target_encode(
                *raw_data[i],
                raw_pub_key=raw_pub_key,
                precision=precision,
            )
            data_encode.append(encode_vector)
            encode_mappings.append(vector_mapping)
        return np.array(data_encode), encode_mappings

    create_pool = False
    if pool is None:
        create_pool = True
        pool = multiprocessing.pool.Pool(num_workers)

    result = pool.starmap(
        functools.partial(_target_encode, raw_pub_key=raw_pub_key, precision=precision),
        raw_data,
    )
    for encode_vector, vector_mapping in result:
        data_encode.append(encode_vector)
        encode_mappings.append(vector_mapping)
    if create_pool:
        pool.close()
    return np.array(data_encode), encode_mappings


def _target_encode(*vector, raw_pub_key, precision):
    n, max_int = raw_pub_key.n, raw_pub_key.max_int
    pos_idxs, neg_idxs = [], []
    pos_exps, neg_exps = [], []
    encode_vector = []

    for i, value in enumerate(vector):
        encode_number = EncodedNumber.encode(raw_pub_key, value, precision=precision)
        encode_vector.append(encode_number)
        encoding = encode_number.encoding
        if n - max_int <= encoding:
            neg_idxs.append(i)
            neg_exps.append(n - encoding)
        else:
            pos_idxs.append(i)
            pos_exps.append(encoding)
    vector_mapping = {
        "pos_idxs": pos_idxs,
        "neg_idxs": neg_idxs,
        "pos_exps": pos_exps,
        "neg_exps": neg_exps,
    }

    return encode_vector, vector_mapping


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
    encrypted_data_ = _crypto_system.encrypt_data(plain_data_, pool=pool_)
    decrypted_data_ = _crypto_system.decrypt_data(encrypted_data_, pool=pool_)
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
