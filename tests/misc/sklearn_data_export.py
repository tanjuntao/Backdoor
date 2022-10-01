import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split


def load_data(name, train):
    if name == 'cancer':  # classification
        cancer = load_breast_cancer()
        x_train, x_test, y_train, y_test = train_test_split(cancer.data,
                                                            cancer.target,
                                                            test_size=0.2,
                                                            random_state=0)

        if train:
            _ids = np.arange(x_train.shape[0])
            _feats = x_train
            _labels = y_train
        else:
            _ids = np.arange(x_train.shape[0],
                             x_train.shape[0] + x_test.shape[0])
            _feats = x_test
            _labels = y_test

    elif name == 'digits':  # classification
        X, Y = load_digits(return_X_y=True)
        odd_idxes = np.where(Y % 2 == 1)[0]
        even_idxes = np.where(Y % 2 == 0)[0]
        Y[odd_idxes] = 1
        Y[even_idxes] = 0
        x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.2,
                                                            random_state=0)
        if train:
            _ids = np.arange(x_train.shape[0])
            _feats = x_train
            _labels = y_train
        else:
            _ids = np.arange(x_train.shape[0],
                             x_train.shape[0] + x_test.shape[0])
            _feats = x_test
            _labels = y_test

    elif name == 'diabetes':  # regression
        # original dataset shape: 442*10
        _whole_feats, _whole_labels = load_diabetes(return_X_y=True,
                                                    scaled=True)
        _n_samples = len(_whole_labels)
        _whole_ids = np.arange(_n_samples)
        test_size = 40  # fixed testing set size
        if train:
            _ids = _whole_ids[:-test_size]
            _feats = _whole_feats[:-test_size]
            _labels = _whole_labels[:-test_size]
        else:
            _ids = _whole_ids[-test_size:]
            _feats = _whole_feats[-test_size:]
            _labels = _whole_labels[-test_size:]

    elif name == 'iris':  # classification, 3 classes
        X, Y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.2,
                                                            random_state=0)
        if train:
            _ids = np.arange(x_train.shape[0])
            _feats = x_train
            _labels = y_train
        else:
            _ids = np.arange(x_train.shape[0], x_train.shape[0] + x_test.shape[0])
            _feats = x_test
            _labels = y_test

    elif name == 'wine':  # classification, 3 classes
        X, Y = load_wine(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.2,
                                                            random_state=0)
        if train:
            _ids = np.arange(x_train.shape[0])
            _feats = x_train
            _labels = y_train
        else:
            _ids = np.arange(x_train.shape[0],
                             x_train.shape[0] + x_test.shape[0])
            _feats = x_test
            _labels = y_test

    else:
        raise ValueError('not supported dataset')

    return _ids, _labels, _feats


if __name__ == '__main__':
    data_names = ['cancer', 'digits', 'diabetes', 'iris', 'wine']

    for name in data_names:
        for train in (True, False):
            ids, labels, feats = load_data(name, train)
            np_dataset = np.concatenate(
                (ids[:, np.newaxis], labels[:, np.newaxis], feats),
                axis=1
            )
            df = pd.DataFrame(np_dataset)
            df[0] = df[0].astype('Int64') # convert id column
            df[1] = df[1].astype('Int64')  # convert label column

            if train:
                df.to_csv('{}-train.csv'.format(name), index=False, header=False)
            else:
                df.to_csv('{}-test.csv'.format(name), index=False, header=False)

    print('Done!')