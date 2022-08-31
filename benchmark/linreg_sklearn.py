import argparse

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.feature import add_intercept, scale


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset in {'diabetes', 'year', 'nyc-taxi'}:
        dataset_name = args.dataset

    else:
        raise ValueError('dataset is not supported.')

    passive_feat_frac = 0.0

    # load dataset
    active_trainset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                   dataset_name=dataset_name,
                                                   train=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=Const.SEQUENCE)
    active_testset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                  dataset_name=dataset_name,
                                                  train=False,
                                                  passive_feat_frac=passive_feat_frac,
                                                  feat_perm_option=Const.SEQUENCE)
    # active_trainset = scale(add_intercept(active_trainset))
    # active_testset = scale(add_intercept(active_testset))
    active_trainset = add_intercept(scale(active_trainset))
    active_testset = add_intercept(scale(active_testset))
    x_train = active_trainset.features
    x_test = active_testset.features
    y_train = active_trainset.labels
    y_test = active_testset.labels

    # initialize model
    regr = LinearRegression()
    # regr = RandomForestRegressor(max_features='1.0', 
    #                              criterion='absolute_error',
    #                              oob_score=True, 
    #                              random_state=99, 
    #                              n_jobs=-1)

    # pipe = make_pipeline(StandardScaler(), regr)

    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    # # start model training
    # regr.fit(x_train, y_train)

    # # evaluate model performance
    # y_pred = regr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('MSE: {:.5f}'.format(mse))
    print('r2: {:.5f}'.format(r2))

