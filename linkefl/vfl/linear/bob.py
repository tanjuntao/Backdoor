import time

import numpy as np
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score
from termcolor import colored

from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.config import LinearConfig as Config
from linkefl.dataio import get_ndarray_dataset as get_dataset
from linkefl.util import sigmoid, save_params


class BobModel:
    def __init__(self, x_train, x_test, y_train, y_test,
                 messenger, cryptosystem, positive_thresh=0.5):
        self.x_train = np.c_[x_train, np.zeros(x_train.shape[0])]
        self.x_test = np.c_[x_test, np.zeros(x_test.shape[0])]
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        self.y_train = y_train
        self.y_test = y_test
        self.params = self._init_weights(x_train.shape[1] + 1)
        self.n_samples = x_train.shape[0]

        self.messenger = messenger
        self.cryptosystem = cryptosystem

        self.POSITIVE_THRESH = positive_thresh

    def _init_weights(self, size):
        if Config.RANDOM_STATE is not None:
            np.random.seed(Config.RANDOM_STATE)
        params = np.random.normal(0, 1.0, size)
        return params

    def _transfer_pubkey(self):
        signal = self.messenger.recv()
        if signal == 'start':
            print('Training protocol started.')
            print('[Bob] Sending public key to Alice...')
            self.messenger.send(self.cryptosystem.pub_key)
            print('[Bob] Done!')
        else:
            raise ValueError('Invalid signal, exit.')

    def _loss_fn(self, y_true, y_pred):
        origin_size = len(y_true)
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                y_true = np.append(y_true, 1)
                y_pred = np.append(y_pred, 1.0)
            else:
                y_true = np.append(y_true, 0)
                y_pred = np.append(y_pred, 0.0)

        return log_loss(y_true=y_true, y_pred=y_pred,
                        normalize=False) / origin_size

    def _loss(self, alice_wx, bob_wx, batch_idxes):
        y_hat = sigmoid(alice_wx + bob_wx)
        train_loss = self._loss_fn(y_true=self.y_train[batch_idxes],
                                   y_pred=y_hat)

        if Config.PENALTY == 'none':
            reg_loss = 0.0
        elif Config.PENALTY == 'l1':
            reg_loss = Config.LAMBDA * abs(self.params).sum()
        elif Config.PENALTY == 'l2':
            reg_loss = 1. / 2 * Config.LAMBDA * (self.params ** 2).sum()
        else:
            raise ValueError('Regularization method not supported now.')

        total_loss = train_loss + reg_loss
        residue = self.y_train[batch_idxes] - y_hat

        return total_loss, residue

    def _grad(self, residue, batch_idxes):
        train_grad = -1 * (
                    residue[:, np.newaxis] * self.x_train[batch_idxes]).mean(
            axis=0)

        if Config.PENALTY == 'none':
            reg_grad = np.zeros(len(self.params))
        elif Config.PENALTY == 'l1':
            reg_grad = Config.LAMBDA * np.sign(self.params)
        elif Config.PENALTY == 'l2':
            reg_grad = Config.LAMBDA * self.params
        else:
            raise ValueError('Regularization method not supported now.')

        return train_grad + reg_grad

    def _gradient_descent(self, grad):
        self.params = self.params - Config.LEARNING_RATE * grad

    def train(self):
        self._transfer_pubkey()

        bs = Config.BATCH_SIZE
        if self.n_samples % bs == 0:
            n_batches = self.n_samples // bs
        else:
            n_batches = self.n_samples // bs + 1

        best_acc, best_auc = 0.0, 0.0
        start_time = None
        compu_time = 0
        for epoch in range(Config.EPOCHS):
            is_best = False
            all_idxes = list(range(self.n_samples))
            batch_losses = []
            for batch in range(n_batches):
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (
                                                                                batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Bob calculates loss and residue
                bob_wx = np.matmul(self.x_train[batch_idxes], self.params)
                alice_wx = self.messenger.recv()
                _begin = time.time()
                if start_time is None:
                    start_time = time.time()
                loss, residue = self._loss(alice_wx, bob_wx, batch_idxes)

                # Bob helps Alice to calcalate gradient
                enc_residue = np.array(
                    self.cryptosystem.encrypt_vector(residue))
                compu_time += time.time() - _begin
                self.messenger.send(enc_residue)
                enc_alice_grad = self.messenger.recv()
                _begin = time.time()
                alice_grad = np.array(
                    self.cryptosystem.decrypt_vector(enc_alice_grad))
                compu_time += time.time() - _begin
                self.messenger.send(alice_grad)

                # Bob calculates its gradient and update model
                bob_grad = self._grad(residue, batch_idxes)
                self._gradient_descent(bob_grad)
                batch_losses.append(loss)

            print(f"\nEpoch: {epoch}, Loss: {np.array(batch_losses).mean()}")

            scores = self.test()
            if scores['acc'] > best_acc:
                best_acc = scores['acc']
                if Config.DATASET_NAME in Config.ACC_DATASETS:
                    is_best = True
            if scores['auc'] > best_auc:
                best_auc = scores['auc']
                if Config.DATASET_NAME in Config.AUC_DATASETS:
                    is_best = True
            print('Acc: {:.5f}, Auc: {:.5f}, f1-score: {:.5f}'.format(
                scores['acc'],
                scores['auc'],
                scores['f1']
            ))
            if is_best:
                save_params(self.params, role='bob')
                print(colored('Best model updates.', 'red'))
            self.messenger.send(is_best)

            # time.sleep(5)

        print(colored('Best history acc: {:.5f}'.format(best_acc), 'red'))
        print(colored('Best history auc: {:.5f}'.format(best_auc), 'red'))
        print(colored('Computation time: {:.5f}'.format(compu_time), 'red'))
        print(colored('Elapsed time: {:.5f}s'.format(time.time() - start_time),
                      'red'))

    def test(self):
        bob_wx = np.matmul(self.x_test, self.params)
        alice_wx = self.messenger.recv()

        probs = sigmoid(alice_wx + bob_wx)
        preds = (probs > self.POSITIVE_THRESH).astype(np.int32)

        accuracy = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds)
        auc = roc_auc_score(self.y_test, probs)

        return {
            'acc': accuracy,
            'f1': f1,
            'auc': auc
        }


# Driven code
if __name__ == '__main__':
    # 1. Load datasets
    X_train, X_test, Y_train, Y_test = get_dataset(
        name=Config.DATASET_NAME,
        role='bob',
        alice_features_frac=Config.ATTACKER_FEATURES_FRAC,
        permutation=Config.PERMUTATION)

    # 2. Initialize cryptosystem
    crypto = crypto_factory()

    # 3. Initialize messenger
    socket = messenger_factory(role='bob')
    print('Bob started, listening...')

    # 4. Initialize model and start training
    bob = BobModel(X_train, X_test, Y_train, Y_test, socket, crypto)
    bob.train()

    # 5. Close messenger, finish training.
    socket.close()



