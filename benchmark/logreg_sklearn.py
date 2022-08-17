import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.feature import add_intercept, scale, parse_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    epochs = 100
    passive_feat_frac = 0.0

    if args.dataset == 'cancer':
        _penalty = 'l2'
        _lambda = 0.01

    elif args.dataset == 'digits':
        _penalty = 'l2'
        _lambda = 0.01

    elif args.dataset == 'epsilon':
        _penalty = 'l2'
        _lambda = 0.01

    elif args.dataset == 'census':
        _penalty = 'l1'
        _lambda = 0.005

    elif args.dataset == 'credit':
        _penalty = 'l2'
        _lambda = 0.001

    elif args.dataset == 'default_credit':
        _penalty = 'l2'
        _lambda = 0.01

    else:
        raise ValueError('dataset is not supported.')


    # load dataset
    active_trainset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                   dataset_name=args.dataset,
                                                   train=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=Const.SEQUENCE)
    active_testset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                  dataset_name=args.dataset,
                                                  train=False,
                                                  passive_feat_frac=passive_feat_frac,
                                                  feat_perm_option=Const.SEQUENCE)
    active_trainset = add_intercept(scale(parse_label(active_trainset)))
    active_testset = add_intercept(scale(parse_label(active_testset)))
    x_train = active_trainset.features
    x_test = active_testset.features
    y_train = active_trainset.labels
    y_test = active_testset.labels

    # initialize classifier
    clf = LogisticRegression(penalty=_penalty,
                             C=1./_lambda,
                             solver='liblinear',
                             max_iter=epochs)
    # start model training
    clf.fit(x_train, y_train)

    # evaluate model performance
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    print('Acc: {:.5f}'.format(acc))
    print('Auc: {:.5f}'.format(auc))
    print('f1: {:.5f}'.format(f1))

