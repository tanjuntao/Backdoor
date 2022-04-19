import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from termcolor import colored


def get_ndarray_dataset(name,
                        role,
                        alice_features_frac,
                        permutation,
                        stats=False,
                        preprocess=True):
    if name == 'breast_cancer':
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

    elif name == 'census_income':
        train_set = np.genfromtxt('../../data/tabular/census_income_train.csv',
                                  delimiter=',')
        test_set = np.genfromtxt('../../data/tabular/census_income_test.csv',
                                 delimiter=',')
        x_train, y_train = train_set[:, 2:], train_set[:, 1]
        x_test, y_test = test_set[:, 2:], test_set[:, 1]
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    elif name == 'give_me_some_credit':
        train_set = np.genfromtxt('../../data/tabular/give_me_some_credit_train.csv',
                                  delimiter=',')
        test_set = np.genfromtxt('../../data/tabular/give_me_some_credit_test.csv',
                                 delimiter=',')
        x_train, y_train = train_set[:, 2:], train_set[:, 1]
        x_test, y_test = test_set[:, 2:], test_set[:, 1]
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    elif name == 'epsilon':
        train_set = np.genfromtxt('../../data/tabular/epsilon_train.csv', delimiter=',')
        test_set = np.genfromtxt('../../data/tabular/epsilon_test.csv', delimiter=',')
        x_train, y_train = train_set[:, 2:], train_set[:, 1]
        x_test, y_test = test_set[:, 2:], test_set[:, 1]
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    elif name == 'pseudo':
        n_train_samples = 100_000
        n_test_samples = 20_000
        n_features = 50
        labels = [0, 1]
        x_train = np.random.rand(n_train_samples, n_features)
        x_test = np.random.rand(n_test_samples, n_features)
        y_train = np.random.choice(labels, size=n_train_samples)
        y_test = np.random.choice(labels, size=n_test_samples)

    else:
        raise ValueError('Not supported now.')

    if stats:
        _dataset_stats(name, x_train, x_test, y_train, y_test)

    if preprocess:
        x_train, x_test = preprocessing.scale(x_train), preprocessing.scale(
            x_test)
    x_train_perm, x_test_perm = x_train[:, permutation], x_test[:, permutation]
    num_alice_features = int(x_train.shape[1] * alice_features_frac)

    assert role in ('alice', 'bob'), "role could only take 'alice' and 'bob'"
    if role == 'alice':
        return x_train_perm[:, :num_alice_features], x_test_perm[:,
                                                     :num_alice_features]
    else:
        return x_train_perm[:, num_alice_features:], x_test_perm[:,
                                                     num_alice_features:], y_train, y_test


def _dataset_stats(name, x_train, x_test, y_train, y_test):
    print('#' * 40)
    print(colored('     Origin Dataset Statistics     ', 'red'))
    print('Dataset name: {}'.format(name))
    print('x_train shape: {}'.format(x_train.shape))
    print('x_test shape: {}'.format(x_test.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('y_test shape: {}'.format(y_test.shape))
    n_pos = (y_train == 1).astype(np.int32).sum()
    n_neg = len(y_train) - n_pos
    print('Positive : Negtive = {} : {}'.format(n_pos, n_neg))
    print('#' * 40)


if __name__ == '__main__':
    # X_train, X_test = get_dataset('digits', 'alice', 0.9, np.random.permutation(64))
    # print(X_train.shape, X_test.shape)
    #
    # X_train, X_test, Y_train, Y_test = get_dataset('digits', 'bob', 0.9, np.random.permutation(64))
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # TODO: import the configuration object here
    pass

    # from config import Config
    #
    # X_train, X_test = get_dataset(name=Config.DATASET_NAME,
    #                               role='bob',
    #                               alice_features_frac=Config.ATTACKER_FEATURES_FRAC,
    #                               permutation=Config.PERMUTATION,
    #                               stats=True,
    #                               preprocess=True)

