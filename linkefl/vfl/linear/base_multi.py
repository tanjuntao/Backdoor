from abc import ABC, abstractmethod
import copy
import datetime
import multiprocessing
import multiprocessing.pool
import os
import random
import time

import numpy as np
from phe import EncodedNumber
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import (
    crypto_factory,
    messenger_factory,
    partial_crypto_factory,
)
from linkefl.config import BaseConfig
from linkefl.crypto import fast_add_ciphers, fast_mul_ciphers
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO


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
                 logger,
                 *,
                 penalty=Const.L2,
                 reg_lambda=0.01,
                 precision=0.001,
                 random_state=None,
                 using_pool=False,
                 num_workers=-1,
                 val_freq=1,
                 saving_model=False,
                 model_path='./models',
                 task='classification',
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
            self.executor_pool = multiprocessing.pool.ThreadPool(num_workers)
            self.scheduler_pool = multiprocessing.pool.ThreadPool(8) # fix n_threads to 8
        else:
            self.executor_pool = None
            self.scheduler_pool = None
        self.val_freq = val_freq
        self.saving_model = saving_model
        self.model_path = model_path
        model_type = Const.VERTICAL_LOGREG if task=='classification' else Const.VERTICAL_LINREG
        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.PASSIVE_NAME,
            model_type=model_type
        )
        self.logger = logger

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
        self.logger.log('[PASSIVE] Requesting publie key...')
        signal = Const.START_SIGNAL
        self.messenger.send(signal)
        public_key = self.messenger.recv()
        self.logger.log('[PASSIVE] Done!')
        return public_key

    @staticmethod
    def _encode(x_train, pub_key, precision):
        x_encode = []
        n_samples = x_train.shape[0]
        for i in range(n_samples):
            row = [EncodedNumber.encode(pub_key, val, precision=precision)
                   for val in x_train[i]]
            x_encode.append(row)
        return np.array(x_encode)

    def _gradient(self, enc_residue, batch_idxes, executor_pool, scheduler_pool):
        # compute gradient of empirical loss term
        if executor_pool is None and scheduler_pool is None:
            enc_train_grad = self._grad(enc_residue, batch_idxes)
        else:
            enc_train_grad = self._grad_pool(
                enc_residue,
                batch_idxes,
                executor_pool,
                scheduler_pool
            )

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

    def _grad_pool(self, enc_residue, batch_idxes, executor_pool, scheduler_pool):
        """compute encrypted gradients manually via Python ThreadPool"""
        if self.crypto_type == Const.PLAIN:
            raise RuntimeError("you should not use pool when crypto type is Plain.")

        n_samples, n_features = len(batch_idxes), getattr(self, 'params').size
        x_encode = getattr(self, 'x_encode')

        # 1. multipy each sample with its corresponding encrypted residue
        enc_train_grads = [None] * n_samples
        data_size = n_samples
        n_schedulers = scheduler_pool._processes
        quotient = data_size // n_schedulers
        remainder = data_size % n_schedulers
        async_results = []
        for idx in range(n_schedulers):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_schedulers - 1:
                end += remainder
            # this will modify enc_train_grads in-place
            result = scheduler_pool.apply_async(
                BaseLinearPassive._target_grad_mul,
                args=(x_encode, enc_residue, batch_idxes, enc_train_grads,
                      start, end, executor_pool)
            )
            async_results.append(result)
        for result in async_results:
            assert result.get() is True

        # 2. transpose this two-dim numpy array so that each row can be averaged to
        #    get the gradient of its corresponding feature
        enc_train_grads = np.array(enc_train_grads).transpose()

        # 3. average the encrypted gradients of each sample to get the final averaged
        #    gradients
        avg_grad = [None] * n_features
        data_size = n_features
        n_schedulers = scheduler_pool._processes
        quotient = data_size // n_schedulers
        remainder = data_size % n_schedulers
        async_results = []
        for idx in range(n_schedulers):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_schedulers - 1:
                end += remainder
            # this will modify avg_grad in-place
            result = scheduler_pool.apply_async(
                BaseLinearPassive._target_grad_add,
                args=(enc_train_grads, avg_grad, start, end, executor_pool)
            )
            async_results.append(result)
        for result in async_results:
            assert result.get() is True

        return np.array(avg_grad)

    @staticmethod
    def _target_grad_mul(x_encode, enc_residue, batch_idxes, enc_train_grads,
                         start, end, executor_pool):
        for k in range(start, end):
            curr_grad = fast_mul_ciphers(
                x_encode[batch_idxes[k]],  # remember to obtain the sample index first
                enc_residue[k],
                executor_pool
            )
            enc_train_grads[k] = curr_grad
        return True

    @staticmethod
    def _target_grad_add(enc_train_grads, avg_grad,
                         start, end, executor_pool):
        batch_size = len(enc_train_grads[start])
        for k in range(start, end):
            grad = fast_add_ciphers(enc_train_grads[k], executor_pool)
            avg_grad[k] = grad * (-1. / batch_size)
        return True

    @staticmethod
    def _mask_grad(enc_grad):
        perm = np.random.permutation(enc_grad.shape[0])
        return enc_grad[perm], perm

    @staticmethod
    def _unmask_grad(masked_grad, perm):
        perm_inverse = np.empty_like(perm)
        perm_inverse[perm] = np.arange(perm.size)
        true_grad = masked_grad[perm_inverse]
        return true_grad

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
                                              num_enc_zeros=10,
                                              gen_from_set=False)
        setattr(self, 'cryptosystem', cryptosystem)

        # encode the training dataset if the crypto type belongs to Paillier family
        if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
            print('encoding dataset...')
            x_encode = BaseLinearPassive._encode(
                getattr(self, 'x_train'),
                public_key,
                self.precision
            )
            print('Done!')
            setattr(self, 'x_encode', x_encode)

        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        commu_plus_compu_time = 0
        # Main Training Loop Here
        self.logger.log('Start collaborative model training...')
        for epoch in range(self.epochs):
            self.logger.log('Epoch: {}'.format(epoch))
            all_idxes = np.arange(n_samples)
            np.random.seed(epoch)
            np.random.shuffle(all_idxes)

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
                enc_grad = self._gradient(
                    enc_residue,
                    batch_idxes,
                    self.executor_pool,
                    self.scheduler_pool
                )
                enc_mask_grad, perm = BaseLinearPassive._mask_grad(enc_grad)
                _begin = time.time()
                self.messenger.send(enc_mask_grad)

                # Receive decrypted masked gradient and update model
                mask_grad = self.messenger.recv()
                true_grad = BaseLinearPassive._unmask_grad(mask_grad, perm)
                commu_plus_compu_time += time.time() - _begin
                self._gradient_descent(getattr(self, 'params'), true_grad)

            # validate the performance of the current model
            if epoch % self.val_freq == 0:
                self.validate(testset)
                is_best = self.messenger.recv()
                if is_best:
                    # save_params(self.params, role=Const.PASSIVE_NAME)
                    self.logger.log('Best model updates.')
                    if self.saving_model:
                        # the use of deepcopy here is to avoid saving other self attrbiutes
                        model_params = copy.deepcopy(getattr(self, 'params'))
                        model_name = self.model_name + "-" + str(trainset.n_samples) + "_samples" + ".model"
                        NumpyModelIO.save(model_params, self.model_path, model_name)

        # after training release the multiprocessing pool
        if self.executor_pool is not None:
            self.executor_pool.close()
            self.scheduler_pool.close()
            self.executor_pool.join()
            self.scheduler_pool.join()

        self.logger.log('Finish model training.')
        self.logger.log('The summation of communication time and computation time is '
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

    @staticmethod
    def online_inference(dataset, model_name, messenger, model_path='./models'):
        assert isinstance(dataset, NumpyDataset), 'inference dataset should be an' \
                                                  'instance of NumpyDataset'
        model_params = NumpyModelIO.load(model_path, model_name)
        wx = np.matmul(dataset.features, model_params)
        messenger.send(wx)
        scores = messenger.recv()
        return scores


class BaseLinearActive(BaseLinear):
    def __init__(self,
                 epochs,
                 batch_size,
                 learning_rate,
                 messenger,
                 cryptosystem,
                 logger,
                 *,
                 penalty=Const.L2,
                 reg_lambda=0.01,
                 crypto_type=Const.PAILLIER,
                 precision=0.001,
                 random_state=None,
                 using_pool=False,
                 num_workers=-1,
                 val_freq=1,
                 saving_model=False,
                 model_path='./models',
                 task='classification',
                 world_size=1
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
        self.world_size = world_size

        if using_pool:
            if num_workers == -1:
                num_workers = os.cpu_count()
            # used to accelerate encrypt_vector (for Paillier only)
            # & decrypt_vector (for both Paillier and FastPaillier)
            self.executor_pool = multiprocessing.pool.ThreadPool(num_workers)
        else:
            self.executor_pool = None
        self.num_workers = num_workers
        self.val_freq = val_freq
        self.saving_model = saving_model
        self.model_path = model_path
        model_type = Const.VERTICAL_LOGREG if task == 'classification' else Const.VERTICAL_LINREG
        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.ACTIVE_NAME,
            model_type=model_type
        )
        self.logger = logger

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
        for id in range(1,self.world_size+1):
            signal = self.messenger.recv(id)
            if signal == Const.START_SIGNAL:
                print('Training protocol started.')
                print('[ACTIVE] Sending public key to passive party...')
                self.messenger.send(self.cryptosystem.pub_key,id)
            else:
                raise ValueError('Invalid signal, exit.')
        print('[ACTIVE] Sending public key done!')

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
