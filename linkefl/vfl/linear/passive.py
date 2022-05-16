import os
from queue import Queue
import threading
import time

import gmpy2
import numpy as np
from phe import EncodedNumber
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import messenger_factory, partial_crypto_factory
from linkefl.config import BaseConfig
from linkefl.dataio import BuildinNumpyDataset
from linkefl.feature import scale
from linkefl.util import save_params


class PassiveLogisticRegression:
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
                 is_multi_thread=False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.messenger = messenger
        self.crypto_type = crypto_type

        self.penalty = penalty
        self.reg_lambda = reg_lambda
        self.precision = precision
        self.random_state = random_state
        self.is_multi_thread = is_multi_thread

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
                   is_multi_thread=config.IS_MULTI_THREAD)

    def _init_weights(self, size):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        else:
            np.random.seed(None)
        params = np.random.normal(0, 1.0, size)
        return params

    def _obtain_pubkey(self):
        print('[PASSIVE] Requesting publie key...')
        signal = Const.START_SIGNAL
        self.messenger.send(signal)
        public_key = self.messenger.recv()
        print('[PASSIVE] Done!')
        return public_key

    def _grad_single_thread(self, enc_residue, batch_idxes):
        if self.crypto_type == Const.PLAIN:
            # using x_train if crypto type is PLAIN
            enc_train_grad = -1 * (enc_residue[:, np.newaxis] *
                                   self.x_train[batch_idxes]).mean(axis=0)
        else:
            # using x_encode if crypto type is Paillier or FastPaillier
            enc_train_grad = -1 * (enc_residue[:, np.newaxis] *
                                   self.x_encode[batch_idxes]).mean(axis=0)

        return enc_train_grad

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
        return res

    def _target_func_grad(self, batches, residues, shared_q):
        gmpy2.get_context().allow_release_gil = True
        enc_grad = -1 * (residues[:, np.newaxis] * self.x_train[batches]).sum(axis=0)
        shared_q.put(enc_grad)

    def _grad(self, enc_residue, batch_idxes):
        # compute gradient of empirical loss term
        if not self.is_multi_thread:
            enc_train_grad = self._grad_single_thread(enc_residue, batch_idxes)
        else:
            # TODO: multi threading to accelerate this operation
            print('start...')
            enc_train_grad = self._grad_multi_thread(enc_residue, batch_idxes)
            print('end \n')
        # print(colored('Gradient time: {}'.format(time.time() - start), 'red'))

        # compute gradient of regularization term
        if self.penalty == Const.NONE:
            reg_grad = np.zeros(len(self.params))
        elif self.penalty == Const.L1:
            reg_grad = self.reg_lambda * np.sign(self.params)
        elif self.penalty == Const.L2:
            reg_grad = self.reg_lambda * self.params
        else:
            raise ValueError('Regularization method not supported now.')
        enc_reg_grad = np.array(self.cryptosystem.encrypt_vector(reg_grad))

        return enc_train_grad + enc_reg_grad

    def _mask_grad(self, enc_grad):
        perm = np.random.permutation(enc_grad.shape[0])

        return enc_grad[perm], perm

    def _unmask_grad(self, masked_grad, perm):
        perm_inverse = np.empty_like(perm)
        perm_inverse[perm] = np.arange(perm.size)
        true_grad = masked_grad[perm_inverse]

        return true_grad

    def _gradient_descent(self, grad):
        self.params = self.params - self.learning_rate * grad

    def _encode(self, x_train, pub_key, precision):
        x_encode = []
        n_samples = x_train.shape[0]

        for i in range(n_samples):
            row = [EncodedNumber.encode(pub_key, val, precision=precision)
                   for val in x_train[i]]
            x_encode.append(row)

        return np.array(x_encode)

    def train(self, x_train, x_val):
        self.x_train = x_train
        self.x_val = x_val
        n_samples = x_train.shape[0]

        # init model parameters
        self.params = self._init_weights(x_train.shape[1])

        # setattr(self, 'x_train', x_train)
        # setattr(self, 'x_val', x_val)
        # setattr(self, 'params', self._init_weights(x_train.shape[1]))
        # setattr(self, 'n_samples', x_train.shape[0])

        # obtain public key from active party and init cryptosystem
        public_key = self._obtain_pubkey()
        self.cryptosystem = partial_crypto_factory(crypto_type=self.crypto_type,
                                                   public_key=public_key,
                                                   num_enc_zeros=10000,
                                                   gen_from_set=False)

        # encode the training dataset if the crypto type belongs to Paillier family
        if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
            self.x_encode = self._encode(self.x_train, public_key, self.precision)

        bs = self.batch_size
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
                wx = np.matmul(self.x_train[batch_idxes], self.params)
                _begin = time.time()
                self.messenger.send(wx)

                # Receive encrypted residue and calculate masked encrypted gradients
                enc_residue = self.messenger.recv()
                commu_plus_compu_time += time.time() - _begin
                enc_grad = self._grad(enc_residue, batch_idxes)
                enc_mask_grad, perm = self._mask_grad(enc_grad)
                _begin = time.time()
                self.messenger.send(enc_mask_grad)

                # Receive decrypted masked gradient and update model
                mask_grad = self.messenger.recv()
                true_grad = self._unmask_grad(mask_grad, perm)
                commu_plus_compu_time += time.time() - _begin
                self._gradient_descent(true_grad)

            # validate the performance of the current model
            is_best = self.validate(self.x_val)
            if is_best:
                # save_params(self.params, role=Const.PASSIVE_NAME)
                print(colored('Best model updates.', 'red'))

        print('The summation of communication time and computation time is '
              '{:.4f}s. In order to obtain the communication time only, you should'
              'substract the computation time which is printed in the console'
              'of active party.'.format(commu_plus_compu_time))

    def validate(self, val_data):
        wx = np.matmul(val_data, self.params)
        self.messenger.send(wx)
        is_best = self.messenger.recv()

        return is_best


if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'digits'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 30000
    _epochs = 200
    _batch_size = 32
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.01
    _random_state = None
    _crypto_type = Const.PLAIN

    # 1. Load datasets
    passive_trainset = BuildinNumpyDataset(dataset_name=dataset_name,
                                           train=True,
                                           role=Const.PASSIVE_NAME,
                                           passive_feat_frac=passive_feat_frac,
                                           feat_perm_option=feat_perm_option)
    passive_testset = BuildinNumpyDataset(dataset_name=dataset_name,
                                          train=False,
                                          role=Const.PASSIVE_NAME,
                                          passive_feat_frac=passive_feat_frac,
                                          feat_perm_option=feat_perm_option)

    # 2. Dataset preprocessing
    passive_trainset = scale(passive_trainset)
    passive_testset = scale(passive_testset)

    # 3. Initialize messenger
    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                  role=Const.PASSIVE_NAME,
                                  active_ip=active_ip,
                                  active_port=active_port,
                                  passive_ip=passive_ip,
                                  passive_port=passive_port)

    # 4. Initialize model and start training
    passive_party = PassiveLogisticRegression(epochs=_epochs,
                                              batch_size=_batch_size,
                                              learning_rate=_learning_rate,
                                              messenger=_messenger,
                                              crypto_type=_crypto_type,
                                              penalty=_penalty,
                                              reg_lambda=_reg_lambda,
                                              random_state=_random_state)

    passive_party.train(passive_trainset.features, passive_testset.features)

    # 5. Close messenger, finish training
    _messenger.close()