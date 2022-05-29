import time

import numpy as np
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.dataio import BuildinNumpyDataset, NumpyDataset
from linkefl.feature import add_intercept, scale, parse_label
from linkefl.feature import ParseLabel, Scale, AddIntercept, Compose
from linkefl.util import sigmoid, save_params
from linkefl.vfl.linear import BaseLinearActive


class ActiveLogReg(BaseLinearActive):
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
                 is_multi_thread=False,
                 val_freq=1,
                 positive_thresh=0.5
    ):
        super(ActiveLogReg, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messenger=messenger,
            cryptosystem=cryptosystem,
            penalty=penalty,
            reg_lambda=reg_lambda,
            crypto_type=crypto_type,
            precision=precision,
            random_state=random_state,
            is_multi_thread=is_multi_thread,
            val_freq=val_freq
        )
        self.POSITIVE_THRESH = positive_thresh

    def _logloss(self, y_true, y_hat):
        origin_size = len(y_true)
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                y_true = np.append(y_true, 1)
                y_hat = np.append(y_hat, 1.0)
            else:
                y_true = np.append(y_true, 0)
                y_hat = np.append(y_hat, 0.0)

        return log_loss(y_true=y_true, y_pred=y_hat, normalize=False) / origin_size

    def _loss(self, y_true, y_hat):
        # Logistic regression uses log-loss as loss function
        train_loss = self._logloss(y_true, y_hat)

        params = getattr(self, 'params')
        if self.penalty == Const.NONE:
            reg_loss = 0.0
        elif self.penalty == Const.L1:
            reg_loss = self.reg_lambda * abs(params).sum()
        elif self.penalty == Const.L2:
            reg_loss = 1. / 2 * self.reg_lambda * (params ** 2).sum()
        else:
            raise ValueError('Regularization method not supported now.')
        total_loss = train_loss + reg_loss

        return total_loss

    def train(self, trainset, testset):
        assert isinstance(trainset, NumpyDataset), 'trainset should be an instance ' \
                                                   'of NumpyDataset'
        assert isinstance(testset, NumpyDataset), 'testset should be an instance' \
                                                  'of NumpyDataset'
        setattr(self, 'x_train', trainset.features)
        setattr(self, 'x_val', testset.features)
        setattr(self, 'y_train', trainset.labels)
        setattr(self, 'y_val', testset.labels)

        # initialize model parameters
        params = self._init_weights(trainset.n_features)
        setattr(self, 'params', params)

        # trainfer public key to passive party
        self._sync_pubkey()

        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        best_acc, best_auc = 0.0, 0.0
        start_time = None
        compu_time = 0
        # Main Training Loop Here
        for epoch in range(self.epochs):
            is_best = False
            all_idxes = list(range(n_samples))
            batch_losses = []
            for batch in range(n_batches):
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Active party calculates loss and residue
                active_wx = np.matmul(getattr(self, 'x_train')[batch_idxes], getattr(self, 'params'))
                passive_wx = self.messenger.recv()
                _begin = time.time()
                if start_time is None:
                    start_time = time.time()
                y_hat = sigmoid(active_wx + passive_wx) # use sigmoid as activation function
                loss = self._loss(getattr(self, 'y_train')[batch_idxes], y_hat)
                residue = self._residue(getattr(self, 'y_train')[batch_idxes], y_hat)

                # Active party helps passive party to calcalate gradient
                enc_residue = np.array(self.cryptosystem.encrypt_vector(residue))
                compu_time += time.time() - _begin
                self.messenger.send(enc_residue)
                enc_passive_grad = self.messenger.recv()
                _begin = time.time()
                passive_grad = np.array(self.cryptosystem.decrypt_vector(enc_passive_grad))
                compu_time += time.time() - _begin
                self.messenger.send(passive_grad)

                # Active party calculates its gradient and update model
                active_grad = self._grad(residue, batch_idxes)
                self._gradient_descent(getattr(self, 'params'), active_grad)
                batch_losses.append(loss)

            # validate model performance
            if epoch % self.val_freq == 0:
                print(f"\nEpoch: {epoch}, Loss: {np.array(batch_losses).mean()}")

                scores = self.validate(testset)
                if scores['acc'] > best_acc:
                    best_acc = scores['acc']
                    is_best = True
                if scores['auc'] > best_auc:
                    best_auc = scores['auc']
                    is_best = True
                print('Acc: {:.5f}, Auc: {:.5f}, f1-score: {:.5f}'.format(
                    scores['acc'],
                    scores['auc'],
                    scores['f1']
                ))
                if is_best:
                    # save_params(self.params, role='bob')
                    print(colored('Best model updates.', 'red'))
                self.messenger.send(is_best)

        print(colored('Best history acc: {:.5f}'.format(best_acc), 'red'))
        print(colored('Best history auc: {:.5f}'.format(best_auc), 'red'))
        print(colored('Computation time: {:.5f}'.format(compu_time), 'red'))
        print(colored('Elapsed time: {:.5f}s'.format(time.time() - start_time), 'red'))

    def validate(self, valset):
        assert isinstance(valset, NumpyDataset), 'valset should be an instance ' \
                                                 'of NumpyDataset'
        active_wx = np.matmul(valset.features, getattr(self, 'params'))
        passive_wx = self.messenger.recv()

        probs = sigmoid(active_wx + passive_wx)
        preds = (probs > self.POSITIVE_THRESH).astype(np.int32)

        accuracy = accuracy_score(valset.labels, preds)
        f1 = f1_score(valset.labels, preds)
        auc = roc_auc_score(valset.labels, probs)

        return {
            'acc': accuracy,
            'f1': f1,
            'auc': auc
        }

    def predict(self, testset):
        return self.validate(testset)


if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'cancer'
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
    _crypto_type = Const.PLAIN
    _random_state = None
    _key_size = 1024

    # 1. Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print('Loading dataset...')
    active_trainset = BuildinNumpyDataset(dataset_name=dataset_name,
                                          train=True,
                                          role=Const.ACTIVE_NAME,
                                          passive_feat_frac=passive_feat_frac,
                                          feat_perm_option=feat_perm_option)
    active_testset = BuildinNumpyDataset(dataset_name=dataset_name,
                                         train=False,
                                         role=Const.ACTIVE_NAME,
                                         passive_feat_frac=passive_feat_frac,
                                         feat_perm_option=feat_perm_option)
    active_trainset = add_intercept(scale(parse_label(active_trainset)))
    active_testset = add_intercept(scale(parse_label(active_testset)))
    print('Done.')

    # Option 2: PyTorch style
    # print('Loading dataset...')
    # transform = Compose([ParseLabel(role=Const.ACTIVE_NAME),
    #                      Scale(role=Const.ACTIVE_NAME),
    #                      AddIntercept(role=Const.ACTIVE_NAME)])
    # active_trainset = BuildinNumpyDataset(dataset_name=dataset_name,
    #                                       train=True,
    #                                       role=Const.ACTIVE_NAME,
    #                                       passive_feat_frac=passive_feat_frac,
    #                                       feat_perm_option=feat_perm_option,
    #                                       transform=transform)
    # active_testset = BuildinNumpyDataset(dataset_name=dataset_name,
    #                                      train=False,
    #                                      role=Const.ACTIVE_NAME,
    #                                      passive_feat_frac=passive_feat_frac,
    #                                      feat_perm_option=feat_perm_option,
    #                                      transform=transform)
    # print('Done.')

    # 3. Initialize cryptosystem
    _crypto = crypto_factory(crypto_type=_crypto_type,
                             key_size=_key_size,
                             num_enc_zeros=10000,
                             gen_from_set=False)

    # 4. Initialize messenger
    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.ACTIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)
    print('ACTIVE PARTY started, listening...')

    # 5. Initialize model and start training
    active_party = ActiveLogReg(epochs=_epochs,
                                batch_size=_batch_size,
                                learning_rate=_learning_rate,
                                messenger=_messenger,
                                cryptosystem=_crypto,
                                penalty=_penalty,
                                reg_lambda=_reg_lambda,
                                random_state=_random_state)

    active_party.train(active_trainset, active_testset)

    # 6. Close messenger, finish training.
    _messenger.close()



