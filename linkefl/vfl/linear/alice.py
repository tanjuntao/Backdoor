import os
from queue import Queue
import threading
import time

import gmpy2
import numpy as np
from phe import EncodedNumber
from termcolor import colored

from linkefl.common.factory import partial_crypto_factory, messenger_factory
from linkefl.config import LinearConfig as Config
from linkefl.dataio import get_ndarray_dataset as get_dataset
from linkefl.util import save_params


class AliceModel:
    def __init__(self, x_train, x_test, messenger):
        self.x_train = x_train
        self.x_test = x_test
        self.params = self._init_weights(x_train.shape[1])
        self.n_samples = x_train.shape[0]

        self.messenger = messenger

    def _init_weights(self, size):
        if Config.RANDOM_STATE is not None:
            np.random.seed(Config.RANDOM_STATE)
        else:
            np.random.seed(None)
        params = np.random.normal(0, 1.0, size)
        return params

    def _obtain_pubkey(self):
        print('[Alice] Requesting publie key...')
        signal = 'start'
        self.messenger.send(signal)
        public_key = self.messenger.recv()
        print('[Alice] Done!')
        return public_key

    def _grad_single_thread(self, enc_residue, batch_idxes):
        if Config.CRYPTO_TYPE == 'plain':
            enc_train_grad = -1 * (enc_residue[:, np.newaxis] *
                                   self.x_train[batch_idxes]).mean(axis=0)
        else:
            enc_train_grad = -1 * (enc_residue[:, np.newaxis] *
                                   self.x_encode[batch_idxes]).mean(axis=0)

        return enc_train_grad

    def _grad_multi_thread(self, enc_residue, batch_idxes, n_threads=None):
        if n_threads is None:
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
        if not Config.MULTI_THREADING:
            enc_train_grad = self._grad_single_thread(enc_residue, batch_idxes)
        else:
            print('start...')
            enc_train_grad = self._grad_multi_thread(enc_residue, batch_idxes)
            print('end \n')
        # print(colored('Gradient time: {}'.format(time.time() - start), 'red'))

        # compute gradient of regularization term
        if Config.PENALTY == 'none':
            reg_grad = np.zeros(len(self.params))
        elif Config.PENALTY == 'l1':
            reg_grad = Config.LAMBDA * np.sign(self.params)
        elif Config.PENALTY == 'l2':
            reg_grad = Config.LAMBDA * self.params
        else:
            raise ValueError('Regularization method not supported now.')
        enc_reg_grad = np.array(self.cryptosystem.encrypt_vector(reg_grad))

        return enc_train_grad + enc_reg_grad

    def _mask_grad(self, enc_grad):
        perm = np.random.permutation(enc_grad.shape[0])

        return enc_grad[perm], perm

    def _gradient_descent(self, mask_grad, perm):
        perm_inverse = np.empty_like(perm)
        perm_inverse[perm] = np.arange(perm.size)
        grad = mask_grad[perm_inverse]

        self.params = self.params - Config.LEARNING_RATE * grad

    def _encode(self, x_train, pub_key, precision=Config.PRECISION):
        x_encode = []
        n_samples = x_train.shape[0]

        for i in range(n_samples):
            row = [EncodedNumber.encode(pub_key, val, precision=precision)
                   for val in x_train[i]]
            x_encode.append(row)

        return np.array(x_encode)

    def train(self):
        public_key = self._obtain_pubkey()
        if Config.CRYPTO_TYPE in ('paillier', 'fast_paillier'):
            self.x_encode = self._encode(self.x_train, public_key)
        self.cryptosystem = partial_crypto_factory(public_key)

        bs = Config.BATCH_SIZE
        if self.n_samples % bs == 0:
            n_batches = self.n_samples // bs
        else:
            n_batches = self.n_samples // bs + 1

        commu_plus_compu_time = 0
        for epoch in range(Config.EPOCHS):
            print('\nEpoch: {}'.format(epoch))
            all_idxes = list(range(self.n_samples))

            for batch in range(n_batches):
                print(f"batch: {batch}")
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Calculate wx and send it to Bob
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
                commu_plus_compu_time += time.time() - _begin
                self._gradient_descent(mask_grad, perm)

            is_best = self.test()
            if is_best:
                save_params(self.params, role='alice')
                print(colored('Best model updates.', 'red'))

        print('The summation of communication time and computation time is '
              '{:.4f}s. In order to obtain the communication time only, you should'
              'substract the computation time which is printed in the console'
              'of Bob.'.format(commu_plus_compu_time))

    def test(self):
        wx = np.matmul(self.x_test, self.params)
        self.messenger.send(wx)
        is_best = self.messenger.recv()

        return is_best


if __name__ == '__main__':
    # 1. Load datasets
    X_train, X_test = get_dataset(name=Config.DATASET_NAME,
                                  role='alice',
                                  alice_features_frac=Config.ATTACKER_FEATURES_FRAC,
                                  permutation=Config.PERMUTATION)

    # 2. Initialize messenger
    socket = messenger_factory(role='alice')

    # 3. Initialize model and start training
    alice = AliceModel(X_train, X_test, socket)
    alice.train()

    # 4. Close messenger, finish training
    socket.close()


