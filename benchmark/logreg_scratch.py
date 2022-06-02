import os
import time
import pickle

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from termcolor import colored


class LogisticRegressionCrossEntropy:
    def __init__(self, penalty='none', lamda=0.1, lr=0.01, fit_intercept=True,
                 max_iter=200, batch_size=32, model_name='lr_cross_entropy',
                 warm_start=True, verbose=True, validate=False,
                 random_state=None, noise=None):

        self.penalty = penalty
        self.lamda = lamda
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model_name = model_name
        self.warm_start = warm_start
        self.verbose = verbose
        self.validate = validate
        self.random_state = random_state
        self.noise = noise
        self.fit_intercept = fit_intercept

        self.w = None
        self.negative_label = 0

    def _init_model_weights(self, size):
        if self.fit_intercept:
            size += 1
        if self.random_state is not None:
            np.random.seed(self.random_state)
        else:
            np.random.seed(None)
        self.w = np.random.normal(0, 1.0, size)

    def _sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))  # avoid overflow runtime error

    def _sigmoid_vec(self, X):
        if isinstance(X, float) or isinstance(X, int):
            return np.array([self._sigmoid(X)])
        else:
            return np.array([self._sigmoid(val) for val in X])

    @staticmethod
    def parse_label(Y):
        # cross entropy loss needs negative label to be 0
        negative_idx = np.where(Y == -1)[0]  # return type: tuple
        Y[negative_idx] = 0
        return Y

    def get_params(self):
        _intercept = self.w[-1] if self.fit_intercept else None
        _coef = self.w[:-1] if self.fit_intercept else self.w

        return {
            "coef": _coef,
            "intercept": _intercept
        }

    def set_params(self, weights):
        self.w = weights

    def set_raw_dataset(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def save_model(self):
        if not os.path.exists('./model'):
            os.mkdir('./model')
        with open('./model/{}.pkl'.format(self.model_name), 'wb') as f:
            pickle.dump(self.w, f)

    @classmethod
    def load_model(cls, path):
        with open(path, mode='rb') as f:
            weights = pickle.load(f)
            lr_instance = cls()
            lr_instance.set_params(weights)

        return lr_instance

    def loss(self, x_train, y_train, batch_idxes):
        # cross entropy loss
        y_hat = self._sigmoid_vec(np.matmul(x_train[batch_idxes], self.w))
        y_pred = np.hstack(((1 - y_hat)[:, np.newaxis], y_hat[:, np.newaxis]))
        train_loss = log_loss(y_train[batch_idxes], y_pred)

        if self.penalty == 'none':
            reg_loss = 0
        elif self.penalty == 'l1':
            reg_loss = self.lamda * abs(self.w).sum()
        elif self.penalty == 'l2':
            reg_loss = 1. / 2 * self.lamda * (self.w ** 2).sum()
        else:
            raise ValueError(
                'Regularization method not supported at this moment')

        return train_loss + reg_loss

    def grad(self, x_train, y_train, batch_idxes):
        # traning data gradient
        y_hat = self._sigmoid_vec(np.matmul(x_train[batch_idxes], self.w))
        r = y_train[batch_idxes] - y_hat

        train_grad = (-1. / len(batch_idxes)) * (
                    r[:, np.newaxis] * x_train[batch_idxes]).sum(axis=0)

        # regularizer gradient
        if self.penalty == 'none':
            reg_grad = np.zeros(x_train.shape[1])
        elif self.penalty == 'l1':
            reg_grad = self.lamda * np.sign(self.w)
        elif self.penalty == 'l2':
            reg_grad = self.lamda * self.w
        else:
            raise ValueError(
                'Regularization method not supported at this moment')

        return train_grad + reg_grad

    def gradient_descent(self, grad):
        self.w = self.w - self.lr * grad

    def fit(self, x_train, y_train):
        train_accs, train_losses, test_accs, test_losses = [], [], [], []

        n_samples, n_features = x_train.shape
        self._init_model_weights(n_features)
        if self.fit_intercept:
            x_train = np.c_[x_train, np.ones(n_samples)]

        # for credit dataset, apply scale after add_intercept
        # for other datasets, apply scale before add_intercept
        # TODO: this can be improved in the future
        x_train = preprocessing.scale(x_train)
        if n_samples % self.batch_size == 0:
            n_batches = n_samples // self.batch_size
        else:
            n_batches = n_samples // self.batch_size + 1

        # load best model
        if self.warm_start and os.path.exists(
                './model/{}.pkl'.format(self.model_name)):
            prev_best_model = LogisticRegressionCrossEntropy.load_model(
                './model/{}.pkl'.format(self.model_name))
            prev_best_scores = prev_best_model.score(self.x_test, self.y_test)
            best_acc, best_auc_score = prev_best_scores['Accuracy'], \
                                       prev_best_scores['auc_score']
            print('Best history model:\nAccuracy: {}, auc_score: {}'.format(
                best_acc, best_auc_score))
        else:
            best_acc, best_auc_score = 0.0, 0.0

        # Start interation
        start = time.time()
        for epoch in range(self.max_iter):
            all_idxes = list(range(n_samples))
            bs = self.batch_size
            batch_losses = []

            for batch in range(n_batches):
                if batch == n_batches - 1:
                    batch_idxes = all_idxes[batch * bs:]
                else:
                    batch_idxes = all_idxes[batch * bs:(batch + 1) * bs]

                loss = self.loss(x_train, y_train, batch_idxes)
                grad = self.grad(x_train, y_train, batch_idxes)

                self.gradient_descent(grad)
                batch_losses.append(loss)

            if self.verbose:
                train_acc = self.score(self.x_train, self.y_train)['Accuracy']
                print('\nEpoch: {}, loss: {}'.format(epoch,
                                                     sum(batch_losses) / len(
                                                         batch_losses)))
                print('Train ACC: {}'.format(train_acc))
                train_accs.append(train_acc)
                train_losses.append(sum(batch_losses) / len(batch_losses))

            if hasattr(self, 'x_test'):
                scores = self.score(self.x_test, self.y_test)
                cur_acc, cur_auc = scores['Accuracy'], scores['auc_score']
                if self.validate:
                    _test_data = np.c_[
                        self.x_test, np.ones(self.x_test.shape[0])]
                    test_loss = self.loss(_test_data, self.y_test,
                                          batch_idxes=np.arange(
                                              self.x_test.shape[0]))
                    print('Test loss: {}'.format(test_loss))
                    print('Test Acc: {}, Test Auc: {}'.format(scores['Accuracy'],
                                                              scores['auc_score']))
                    test_accs.append(scores['Accuracy'])
                    test_losses.append(test_loss)

                if (cur_acc > best_acc and cur_auc > best_auc_score) or (
                        cur_acc - best_acc >= 0.01) or (
                        cur_auc - best_auc_score >= 0.01):
                    best_acc, best_auc_score = scores['Accuracy'], scores['auc_score']
                    print(colored('Best model updated.', 'red'))

        return best_acc, best_auc_score

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        predictions = [1 if prob > threshold else self.negative_label for prob
                       in probs]

        return np.array(predictions)

    def predict_proba(self, X):
        if self.fit_intercept and len(X.shape) == 2:
            X = np.c_[X, np.ones(X.shape[0])]
        if self.fit_intercept and len(X.shape) == 1:
            X = np.append(X, 1)
        X = preprocessing.scale(X)
        digits = np.matmul(X, self.w)
        probs = self._sigmoid_vec(digits)

        return probs

    def score(self, X, Y):
        predictions = self.predict(X)
        probs = self.predict_proba(X)

        acc_ = (predictions == Y).astype(int).mean()
        f1_score_ = f1_score(Y, predictions)
        auc_score_ = roc_auc_score(Y, probs)

        return {
            "Accuracy": acc_,
            "f1-score": f1_score_,
            "auc_score": auc_score_,
        }


def get_dataset(name):
    curr_path = os.path.abspath(os.path.dirname(__file__))
    if name == 'cancer':
        cancer = load_breast_cancer()
        x_train, x_test, y_train, y_test = train_test_split(cancer.data,
                                                            cancer.target,
                                                            test_size=0.2,
                                                            random_state=0)

    elif name == 'digits':
        X, Y = load_digits(return_X_y=True)
        odd_idxes, even_idxes = np.where(Y % 2 == 1)[0], np.where(Y % 2 == 0)[0]
        Y[odd_idxes] = 1
        Y[even_idxes] = 0
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                            random_state=0)

    elif name == 'census':
        train_path = os.path.join(curr_path, '../LinkeFL/data/tabular/census_income_train.csv')
        test_path = os.path.join(curr_path, '../LinkeFL/data/tabular/census_income_test.csv')
        train_set = np.genfromtxt(train_path, delimiter=',')
        test_set = np.genfromtxt(test_path, delimiter=',')
        x_train, y_train = train_set[:, 2:], train_set[:, 1]
        x_test, y_test = test_set[:, 2:], test_set[:, 1]
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)


    elif name == 'credit':
        train_path = os.path.join(curr_path,
                                  '../LinkeFL/data/tabular/give_me_some_credit_train.csv')
        test_path = os.path.join(curr_path,
                                 '../LinkeFL/data/tabular/give_me_some_credit_test.csv')
        train_set = np.genfromtxt(train_path, delimiter=',')
        test_set = np.genfromtxt(test_path, delimiter=',')
        x_train, y_train = train_set[:, 2:], train_set[:, 1]
        x_test, y_test = test_set[:, 2:], test_set[:, 1]
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
        print(train_set.shape[0] + test_set.shape[0], train_set.shape[1])

    elif name == 'epsilon':
        train_path = os.path.join(curr_path,
                                  '../LinkeFL/data/tabular/epsilon_train.csv')
        test_path = os.path.join(curr_path,
                                 '../LinkeFL/data/tabular/epsilon_test.csv')
        train_set = np.genfromtxt(train_path, delimiter=',')
        test_set = np.genfromtxt(test_path, delimiter=',')
        x_train, y_train = train_set[:, 2:], train_set[:, 1]
        x_test, y_test = test_set[:, 2:], test_set[:, 1]
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    else:
        raise ValueError('Not supported now')

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    clf = LogisticRegressionCrossEntropy(penalty='l2',
                                         lamda=0.01,
                                         lr=0.01,
                                         max_iter=200,
                                         batch_size=32,
                                         warm_start=False,
                                         random_state=None,
                                         verbose=True,
                                         validate=True,
                                         noise=None)

    x_train, x_test, y_train, y_test = get_dataset(name='census')

    clf.set_raw_dataset(x_train, x_test, y_train, y_test)

    start = time.time()
    best_acc, best_auc = clf.fit(x_train, y_train)
    print('Training time: {:.5f}'.format(time.time() - start))
    print('best acc: {:.5f}'.format(best_acc))
    print('best auc: {:.5f}'.format(best_auc))
    print(clf.score(x_test, y_test))

