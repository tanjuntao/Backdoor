import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.feature import add_intercept


if __name__ == '__main__':
    dataset_name = 'diabetes'
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
    active_trainset = add_intercept(active_trainset)
    active_testset = add_intercept(active_testset)
    x_train = active_trainset.features
    x_test = active_testset.features
    y_train = active_trainset.labels
    y_test = active_testset.labels

    # initialize model
    regr = LinearRegression()

    # start model training
    regr.fit(x_train, y_train)

    # evaluate model performance
    y_pred = regr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('MSE: {:.5f}'.format(mse))
    print('r2: {:.5f}'.format(r2))

