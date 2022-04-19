from abc import ABC, abstractmethod
import multiprocessing
import os
import warnings

import numpy as np
import torch


class CryptoSystem:
    """Base class of cryptosystem"""
    def __init__(self, key_size=1024):
        """Initialize a cryptosystem.

        Args:
            key_size: Key size of cryptosystem, default 1024 bits.
        """
        self.key_size = key_size

    @abstractmethod
    def __gen_key(self):
        """Generate public key and private key."""
        pass

    @abstractmethod
    def encrypt(self, plaintext):
        """Encrypt single plaintext message."""
        pass

    @abstractmethod
    def decrypt(self, ciphertext):
        """Decrypt single ciphertext"""
        pass

    @abstractmethod
    def encrypt_vector(self, plain_vector, using_mp=False, n_processes=None):
        """Encrypt a vector with more than one plaintext message"""
        # TODO: using multi threading
        pass

    @abstractmethod
    def decrypt_vector(self, cipher_vector, using_mp=False, n_processes=None):
        """Decrypt a vector of ciphertext"""
        # TODO: using multi theading
        pass

    def params_checking(self, vector, using_mp, n_processes):
        """Check parameters"""
        if type(vector) not in (list, np.ndarray, torch.Tensor):
            raise TypeError('vector could only be type of list, '
                            'numpy.ndarray and torch.Tensor, but {}'
                            'got'.format(type(vector)))

        if using_mp and len(vector) < 1500:
            warnings.warn('It is not recommended to use multiprocessing'
                          'when length of vector is less than 1500.')

        if n_processes != os.cpu_count():
            warnings.warn('It is recommended to set n_processes equals to '
                          'number of CPUs on the computer.')

    # TODO: add multiprocessing.Pool
    def func_mp(self, op, vector, n_processes):
        """Using python multiprocessing to accelerate op opration on vector
        using n_processes process.

        Args:
            op [funcion]: A python function.
            vector [list]: Data which needs to be operated by op.
            n_processes [int]: How many processes to use when using multiprocess.

        Returns:
            Data after operation.
        """
        manager = multiprocessing.Manager()
        _shared_data = manager.list(vector)

        length = len(vector) // n_processes
        remainder = len(vector) % n_processes
        process_list = []
        for idx in range(n_processes):
            _start = idx * length
            _end = (idx + 1) * length
            if idx == n_processes - 1:
                _end += remainder
            p = multiprocessing.Process(target=CryptoSystem._target_func,
                                        args=(_shared_data, _start, _end, op))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

        return [item for item in _shared_data]

    @staticmethod
    def _target_func(shared_data, start, end, op):
        for i in range(start, end):
            shared_data[i] = op(shared_data[i])

