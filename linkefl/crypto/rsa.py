import functools
import multiprocessing
import os
from pathlib import Path
import threading

from Crypto.PublicKey import RSA
import gmpy2

from linkefl.config import BaseConfig


def _target_mp_manager(X, _start, _end, d, n):
    for i in range(_start, _end):
        X[i] = gmpy2.powmod(X[i], d, n)


def _target_mp_pool(x, d, n):
    return gmpy2.powmod(x, d, n)


class RSACrypto:
    # TODO: inherited from base crypto class
    SECRET_CODE = 'linkefl'
    PRIV_KEY_NAME = 'rsa_priv_key.bin'
    PROJECT_DIR = '.linkefl'

    def __init__(self, key_size=1024, e=0x10001, private_key=None):
        if private_key is None:
            self.priv_key = RSA.generate(key_size, e=e)
            self._save_key(self.priv_key)
        else:
            self.priv_key = private_key

        self.pub_key = RSA.construct((self.priv_key.n, self.priv_key.e))
        self.d = self.priv_key.d
        self.n = self.priv_key.n
        self.e = self.priv_key.e

    def _save_key(self, key):
        encrypted_key = key.export_key(passphrase=RSACrypto.SECRET_CODE,
                                       pkcs=8,
                                       protection='scryptAndAES128-CBC')
        target_dir = os.path.join(Path.home(), RSACrypto.PROJECT_DIR)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        full_path = os.path.join(target_dir, RSACrypto.PRIV_KEY_NAME)
        with open(full_path, 'wb') as f:
            f.write(encrypted_key)

    @classmethod
    def from_config(cls, config):
        assert isinstance(config, BaseConfig), 'config object should be an ' \
                                               'instance of BaseConfig class.'
        return cls(key_size=config.KEY_SIZE, e=config.PUB_E)

    @classmethod
    def from_private(cls):
        full_path = os.path.join(Path.home(), cls.PROJECT_DIR, cls.PRIV_KEY_NAME)

        # check if the private key exists
        if not os.path.exists(full_path):
            raise FileNotFoundError('There is no RSA private key within the'
                                    '~/.linkefl/ directory.')

        # load the private key
        encoded_key = open(full_path, 'rb').read()
        private_key = RSA.import_key(encoded_key, passphrase=cls.SECRET_CODE)

        # delete the private key
        os.remove(full_path)

        return cls(private_key=private_key)

    def encrypt(self, plaintext):
        return gmpy2.powmod(plaintext, self.e, self.n)

    def decrypt(self, ciphertext):
        return gmpy2.powmod(ciphertext, self.d, self.n)

    def sign(self, x):
        return self.decrypt(x)

    def sign_set(self, X):
        return [self.sign(x) for x in X]

    def inverse(self, x):
        return gmpy2.invert(x, self.n)

    def mulmod(self, x, y, n):
        """return (x * y) mod n"""
        # If the data type of all three arguments of mulmod() is mpz, then
        # the performance is better.
        if type(x) == int:
            x = gmpy2.mpz(x)
        if type(y) == int:
            y = gmpy2.mpz(y)
        if type(n) == int:
            n = gmpy2.mpz(n)
        return gmpy2.mod(gmpy2.mul(x, y), n)

    def sign_set_mp_manager(self, X, n_processes=-1):
        if n_processes == -1:
            n_processes = os.cpu_count()

        manager = multiprocessing.Manager()
        shared_data = manager.list(X)

        # store these two self variables locally
        d = self.d
        n = self.n
        quotient = len(X) // n_processes
        remainder = len(X) % n_processes
        processes = []
        for idx in range(n_processes):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_processes - 1:
                end += remainder
            # You cannot place the target function within the python class,
            # this is because the pub_key and priv_key attrbiutes is not
            # pickleable, which causes that the target function cannot passed
            # to sub-processes from parent process.
            # In one word, the target function and the corresponding arguments
            # should not contain any self.methods or self.attributes.
            p = multiprocessing.Process(target=_target_mp_manager,
                                        args=(shared_data, start, end, d, n))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # shared_data is an proxy object, you cannot
        # directly return it to the caller. Instead, you need to retrive the
        # actual values behind this proxy object.
        return [item for item in shared_data]

    def sign_set_mp_pool(self, X, n_processes=-1):
        if n_processes == -1:
            n_processes = os.cpu_count()

        manager = multiprocessing.Manager()
        shared_data = manager.list(X)

        d = self.d
        n = self.n
        with multiprocessing.Pool(n_processes) as p:
            # the gmpy2.powmod function cannot accept keyword arguments,
            # so you need to write a wrapper function of it.
            # Note that the target function can also not be within this python class
            res = p.map(functools.partial(_target_mp_pool, d=d, n=n),
                        shared_data)

        return res

    def sign_set_thread(self, X, n_threads=-1):
        if n_threads == -1:
            n_threads = os.cpu_count()

        release_gil = True
        quotient = len(X) // n_threads
        remainder = len(X) % n_threads
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
            t = threading.Thread(target=self._target_sign_thread,
                                 args=(X, start, end, release_gil))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        return X

    def encrypt_set_thread(self, X, n_threads=-1):
        if n_threads == -1:
            n_threads = os.cpu_count()

        release_gil = True
        e = self.e
        n = self.n
        quotient = len(X) // n_threads
        remainder = len(X) % n_threads
        threads = []
        for idx in range(n_threads):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_threads - 1:
                end += remainder
            t = threading.Thread(target=self._target_encrypt_thread,
                                 args=(X, start, end, e, n, release_gil))
            threads.append(t)
        for t in threads:
            t.join()

        return X

    def _target_sign_thread(self, X, start, end, release_gil=True):
        # You only need to release the GIL, the re-acquisition of it is
        # automaticlly done by gmpy2.
        gmpy2.get_context().allow_release_gil = release_gil

        for i in range(start, end):
            X[i] = self.sign(X[i])

    def _target_encrypt_thread(self, X, start, end, e, n, release_gil=True):
        gmpy2.get_context().allow_release_gil = release_gil
        sub_list = X[start:end]
        X[start:end] = gmpy2.powmod_list(sub_list, e, n)


class PartialRSACrypto:
    # TODO: inherited from RSACrypto
    def __init__(self, pub_key):
        self.pub_key = pub_key
        self.e = self.pub_key.e
        self.n = self.pub_key.n

    def encrypt(self, plaintext):
        return gmpy2.powmod(plaintext, self.e, self.n)

    def inverse(self, x):
        return gmpy2.invert(x, self.n)

    def mulmod(self, x, y, n):
        if type(x) == int:
            x = gmpy2.mpz(x)
        if type(y) == int:
            y = gmpy2.mpz(y)
        if type(n) == int:
            n = gmpy2.mpz(n)
        return gmpy2.mod(gmpy2.mul(x, y), n)

    def _target_encrypt_thread(self, X, start, end, e, n, release_gil=True):
        gmpy2.get_context().allow_release_gil = release_gil
        sub_list = X[start:end]
        X[start:end] = gmpy2.powmod_list(sub_list, e, n)

    def encrypt_set_thread(self, X, n_threads=-1):
        if n_threads == -1:
            n_threads = os.cpu_count()

        release_gil = True
        e = self.e
        n = self.n
        quotient = len(X) // n_threads
        remainder = len(X) % n_threads
        threads = []
        for idx in range(n_threads):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_threads - 1:
                end += remainder
            t = threading.Thread(target=self._target_encrypt_thread,
                                 args=(X, start, end, e, n, release_gil))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        return X


if __name__ == '__main__':
    crypto = RSACrypto()
    import time

    start_time = time.time()
    crypto.sign_set_mp_manager([i for i in range(100_000)])
    print('Elapsed time: {}'.format(time.time() - start_time))
