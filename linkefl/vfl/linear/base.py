from abc import ABC, abstractmethod
import multiprocessing
import os
from queue import Queue
import threading
import time

import gmpy2
import numpy as np
from phe import EncodedNumber, EncryptedNumber
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import (
    messenger_factory,
    crypto_factory,
    partial_crypto_factory
)
from linkefl.config import BaseConfig
from linkefl.dataio import NumpyDataset
from linkefl.util import save_params


def _target_grad(*args):
    return sum(args)


class BaseLinear(ABC):
    def __init__(self, learning_rate, random_state):
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _init_weights(self, size):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        else:
            np.random.seed(None)
        params = np.random.normal(0, 1.0, size)
        return params

    def _gradient_descent(self, params, grad):
        params -= self.learning_rate * grad

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError('should not call abstract class method')

    @abstractmethod
    def _sync_pubkey(self):
        pass

    @abstractmethod
    def _grad(self, residue, batch_idxes):
        pass

    @abstractmethod
    def train(self, trainset, testset):
        pass

    @abstractmethod
    def validate(self, valset):
        pass

    @abstractmethod
    def predict(self, testset):
        pass


class BaseLinearPassive(BaseLinear):
    def __init__(self,
                 epochs,
                 batch_size,
                 learning_rate,
                 messenger,
                 crypto_type,
                 *,
                 penalty=Const.L2,
                 reg_lambda=0.01,
                 precision=0.001,
                 random_state=None,
                 using_pool=False,
                 num_workers=-1,
                 val_freq=1
    ):
        super(BaseLinearPassive, self).__init__(learning_rate, random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.messenger = messenger
        self.crypto_type = crypto_type

        self.penalty = penalty
        self.reg_lambda = reg_lambda
        self.precision = precision
        self.random_state = random_state
        self.using_pool = using_pool
        if using_pool:
            if num_workers == -1:
                num_workers = os.cpu_count()
            self.pool = multiprocessing.Pool(num_workers)
        else:
            self.pool = None
        self.val_freq = val_freq

    @classmethod
    def from_config(cls, config):
        assert isinstance(config, BaseConfig), 'config object should be an ' \
                                               'instance of BaseConfig class.'
        # initialize messenger
        messenger = messenger_factory(messenger_type=config.MESSENGER_TYPE,
                                      role=config.ROLE,
                                      active_ip=config.ACTIVE_IP,
                                      active_port=config.ACTIVE_PORT,
                                      passive_ip=config.PASSIVE_IP,
                                      passive_port=config.PASSIVE_PORT,
                                      verbose=config.VERBOSE)

        return cls(epochs=config.EPOCHS,
                   batch_size=config.BATCH_SIZE,
                   learning_rate=config.LEARNING_RATE,
                   messenger=messenger,
                   crypto_type=config.CRYPTO_TYPE,
                   penalty=config.PENALTY,
                   reg_lambda=config.REG_LAMBDA,
                   precision=config.PRECISION,
                   random_state=config.RANDOM_STATE,
                   using_pool=config.USING_POOL,
                   num_workers=config.NUM_WORKERS,
                   val_freq=config.VAL_FREQ)

    def _sync_pubkey(self):
        print('[PASSIVE] Requesting publie key...')
        signal = Const.START_SIGNAL
        self.messenger.send(signal)
        public_key = self.messenger.recv()
        print('[PASSIVE] Done!')
        return public_key

    def _grad(self, enc_residue, batch_idxes):
        if self.crypto_type == Const.PLAIN:
            # using x_train if crypto type is PLAIN
            enc_train_grad = -1 * (enc_residue[:, np.newaxis] *
                                   getattr(self, 'x_train')[batch_idxes]).mean(axis=0)
        else:
            # using x_encode if crypto type is Paillier or FastPaillier
            enc_train_grad = -1 * (enc_residue[:, np.newaxis] *
                                   getattr(self, 'x_encode')[batch_idxes]).mean(axis=0)

        return enc_train_grad

    def _grad_mp_pool(self, enc_residue, batch_idxes, worker_pool):
        """compute encrypted gradients manully"""
        n_samples, n_features = len(batch_idxes), getattr(self, 'params').size
        pub_key = getattr(self, 'cryptosystem').pub_key
        n = pub_key.n
        n_squared = pub_key.n ** 2
        max_int = pub_key.max_int
        r_ciphers = [enc_r.ciphertext(False) for enc_r in enc_residue]
        r_ciphers_neg = [gmpy2.invert(r_cipher, n_squared) for r_cipher in r_ciphers]

        # collect encrypted gradient items
        enc_train_grads = [[] for _ in range(n_features)]
        for i in range(n_samples):
            for j in range(n_features):
                row = getattr(self, 'x_encode')[batch_idxes[i]]
                encoding = row[j].encoding
                exponent = row[j].exponent + enc_residue[i].exponent
                if n - max_int < encoding:
                    ciphertext = gmpy2.powmod(r_ciphers_neg[i], n - encoding, n_squared)
                else:
                    ciphertext = gmpy2.powmod(r_ciphers[i], encoding, n_squared)
                enc_j = EncryptedNumber(pub_key, ciphertext, exponent)
                enc_train_grads[j].append(enc_j)

        # using pool to add encrypted gradients in parallel
        avg_grads = worker_pool.starmap(_target_grad, enc_train_grads)

        # average gradients
        for j in range(n_features):
            avg_grads[j] = avg_grads[j] * (-1. / len(batch_idxes))

        # convert grads from python list to numpy.ndarray
        avg_grads = np.array(avg_grads)
        return avg_grads

    def _grad_multi_thread(self, enc_residue, batch_idxes, n_threads=-1):
        if n_threads == -1:
            n_threads = os.cpu_count()

        bs = len(batch_idxes)
        quotient = bs // n_threads
        remainder = bs % n_threads
        threads = []
        shared_q = Queue()
        for i in range(n_threads):
            start = i * quotient
            end = (i + 1) * quotient
            if i == n_threads - 1:
                end += remainder
            t = threading.Thread(target=self._target_func_grad,
                                 args=(batch_idxes[start:end],
                                       enc_residue[start:end],
                                       shared_q))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        res = shared_q.get()
        while not shared_q.empty():
            res += shared_q.get()
        return res / bs # average gradients

    def _target_func_grad(self, batches, residues, shared_q):
        gmpy2.get_context().allow_release_gil = True
        enc_grad = -1 * (residues[:, np.newaxis] * getattr(self, 'x_train')[batches]).sum(axis=0)
        shared_q.put(enc_grad)

    def _gradient(self, enc_residue, batch_idxes, worker_pool):
        # compute gradient of empirical loss term
        if worker_pool is None:
            enc_train_grad = self._grad(enc_residue, batch_idxes)
        else:
            enc_train_grad = self._grad_mp_pool(enc_residue, batch_idxes, worker_pool)

        # compute gradient of regularization term
        params = getattr(self, 'params')
        if self.penalty == Const.NONE:
            reg_grad = np.zeros(len(params))
        elif self.penalty == Const.L1:
            reg_grad = self.reg_lambda * np.sign(params)
        elif self.penalty == Const.L2:
            reg_grad = self.reg_lambda * params
        else:
            raise ValueError('Regularization method not supported now.')
        enc_reg_grad = np.array(getattr(self, 'cryptosystem').encrypt_vector(reg_grad))

        return enc_train_grad + enc_reg_grad

    def _mask_grad(self, enc_grad):
        perm = np.random.permutation(enc_grad.shape[0])
        return enc_grad[perm], perm

    def _unmask_grad(self, masked_grad, perm):
        perm_inverse = np.empty_like(perm)
        perm_inverse[perm] = np.arange(perm.size)
        true_grad = masked_grad[perm_inverse]
        return true_grad

    def _encode(self, x_train, pub_key, precision):
        x_encode = []
        n_samples = x_train.shape[0]
        for i in range(n_samples):
            row = [EncodedNumber.encode(pub_key, val, precision=precision)
                   for val in x_train[i]]
            x_encode.append(row)
        return np.array(x_encode)

    def train(self, trainset, testset):
        assert isinstance(trainset, NumpyDataset), 'trainset should be an instance ' \
                                                   'of NumpyDataset'
        assert isinstance(testset, NumpyDataset), 'testset should be an instance' \
                                                  'of NumpyDataset'
        setattr(self, 'x_train', trainset.features)
        setattr(self, 'x_val', testset.features)
        # init model parameters
        params = self._init_weights(trainset.n_features)
        setattr(self, 'params', params)

        # obtain public key from active party and init cryptosystem
        public_key = self._sync_pubkey()
        cryptosystem = partial_crypto_factory(crypto_type=self.crypto_type,
                                              public_key=public_key,
                                              num_enc_zeros=10000,
                                              gen_from_set=False)
        setattr(self, 'cryptosystem', cryptosystem)

        # encode the training dataset if the crypto type belongs to Paillier family
        if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
            x_encode = self._encode(getattr(self, 'x_train'), public_key, self.precision)
            setattr(self, 'x_encode', x_encode)

        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        commu_plus_compu_time = 0
        # Main Training Loop Here
        for epoch in range(self.epochs):
            print('\nEpoch: {}'.format(epoch))
            all_idxes = list(range(n_samples))

            for batch in range(n_batches):
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Calculate wx and send it to active party
                wx = np.matmul(getattr(self, 'x_train')[batch_idxes], getattr(self, 'params'))
                _begin = time.time()
                self.messenger.send(wx)

                # Receive encrypted residue and calculate masked encrypted gradients
                enc_residue = self.messenger.recv()
                commu_plus_compu_time += time.time() - _begin
                enc_grad, add_time, _powmod_time = self._gradient(enc_residue, batch_idxes, self.pool)
                enc_mask_grad, perm = self._mask_grad(enc_grad)
                _begin = time.time()
                self.messenger.send(enc_mask_grad)

                # Receive decrypted masked gradient and update model
                mask_grad = self.messenger.recv()
                true_grad = self._unmask_grad(mask_grad, perm)
                commu_plus_compu_time += time.time() - _begin
                self._gradient_descent(getattr(self, 'params'), true_grad)

            # validate the performance of the current model
            if epoch % self.val_freq == 0:
                self.validate(testset)
                is_best = self.messenger.recv()
                if is_best:
                    # save_params(self.params, role=Const.PASSIVE_NAME)
                    print(colored('Best model updates.', 'red'))

        # after training release the multiprocessing pool
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

        print('The summation of communication time and computation time is '
              '{:.4f}s. In order to obtain the communication time only, you should'
              'substract the computation time which is printed in the console'
              'of active party.'.format(commu_plus_compu_time))

    def validate(self, valset):
        assert isinstance(valset, NumpyDataset), 'valset should be an instance ' \
                                                 'of NumpyDataset'
        wx = np.matmul(valset.features, getattr(self, 'params'))
        self.messenger.send(wx)

    def predict(self, testset):
        self.validate(testset)


class BaseLinearActive(BaseLinear):
    def __init__(self,
                 epochs,
                 batch_size,
                 learning_rate,
                 messenger,
                 cryptosystem,
                 *,
                 penalty=Const.L2,
                 reg_lambda=0.01,
                 crypto_type=Const.PAILLIER,
                 precision=0.001,
                 random_state=None,
                 using_pool=False,
                 num_workers=-1,
                 val_freq=1
    ):
        super(BaseLinearActive, self).__init__(learning_rate, random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.messenger = messenger
        self.cryptosystem = cryptosystem

        self.penalty = penalty
        self.reg_lambda = reg_lambda
        self.crypto_type = crypto_type
        self.precision = precision
        self.random_state = random_state
        self.using_pool = using_pool
        self.num_workers = num_workers
        self.val_freq = val_freq

    @classmethod
    def from_config(cls, config):
        assert isinstance(config, BaseConfig), 'config object should be an ' \
                                               'instance of BaseConfig class.'
        # initialize messenger
        messenger = messenger_factory(messenger_type=config.MESSENGER_TYPE,
                                      role=config.ROLE,
                                      active_ip=config.ACTIVE_IP,
                                      active_port=config.ACTIVE_PORT,
                                      passive_ip=config.PASSIVE_IP,
                                      passive_port=config.PASSIVE_PORT,
                                      verbose=config.VERBOSE)
        # initialize cryptosystem
        crypto = crypto_factory(crypto_type=config.CRYPTO_TYPE,
                                key_size=config.KEY_SIZE,
                                num_enc_zeros=config.NUM_ENC_ZEROS,
                                gen_from_set=config.GEN_FROM_SET)

        return cls(epochs=config.EPOCHS,
                   batch_size=config.BATCH_SIZE,
                   learning_rate=config.LEARNING_RATE,
                   messenger=messenger,
                   cryptosystem=crypto,
                   penalty=config.PENALTY,
                   reg_lambda=config.REG_LAMBDA,
                   crypto_type=config.CRYPTO_TYPE,
                   precision=config.PRECISION,
                   random_state=config.RANDOM_STATE,
                   is_multi_thread=config.IS_MULTI_THREAD)

    def _sync_pubkey(self):
        signal = self.messenger.recv()
        if signal == Const.START_SIGNAL:
            print('Training protocol started.')
            print('[ACTIVE] Sending public key to passive party...')
            self.messenger.send(self.cryptosystem.pub_key)
            print('[ACTIVE] Done!')
        else:
            raise ValueError('Invalid signal, exit.')

    def _residue(self, y_true, y_hat):
        return y_true - y_hat

    def _grad(self, residue, batch_idxes):
        if not hasattr(self, 'x_train'):
            raise AttributeError('x_train is not added to this object')
        if not hasattr(self, 'params'):
            raise AttributeError('params is not added to this object')

        train_grad = -1 * (residue[:, np.newaxis] * getattr(self, 'x_train')[batch_idxes]).mean(axis=0)
        params = getattr(self, 'params')
        if self.penalty == Const.NONE:
            reg_grad = np.zeros(len(params))
        elif self.penalty == Const.L1:
            reg_grad = self.reg_lambda * np.sign(params)
        elif self.penalty == Const.L2:
            reg_grad = self.reg_lambda * params
        else:
            raise ValueError('Regularization method not supported now.')

        return train_grad + reg_grad

    def train(self, trainset, testset):
        raise NotImplementedError('should not call this method within base class')

    def validate(self, valset):
        raise NotImplementedError('should not call this method within base class')

    def predict(self, testset):
        raise NotImplementedError('should not call this method within base class')