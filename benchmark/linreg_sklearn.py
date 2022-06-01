import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from linkefl.common.const import Const
from linkefl.dataio import BuildinNumpyDataset
from linkefl.feature import add_intercept


if __name__ == '__main__':
    dataset_name = 'diabetes'

    # load dataset
    active_trainset = BuildinNumpyDataset(dataset_name=dataset_name,
                                          train=True,
                                          role=Const.ACTIVE_NAME,
                                          passive_feat_frac=0.5,
                                          feat_perm_option=Const.SEQUENCE)
    active_testset = BuildinNumpyDataset(dataset_name=dataset_name,
                                         train=False,
                                         role=Const.ACTIVE_NAME,
                                         passive_feat_frac=0.5,
                                         feat_perm_option=Const.SEQUENCE)
    passive_trainset = BuildinNumpyDataset(dataset_name=dataset_name,
                                          train=True,
                                          role=Const.PASSIVE_NAME,
                                          passive_feat_frac=0.5,
                                          feat_perm_option=Const.SEQUENCE)
    passive_testset = BuildinNumpyDataset(dataset_name=dataset_name,
                                         train=False,
                                         role=Const.PASSIVE_NAME,
                                         passive_feat_frac=0.5,
                                         feat_perm_option=Const.SEQUENCE)
    active_trainset = add_intercept(active_trainset)
    active_testset = add_intercept(active_testset)
    x_train = np.concatenate((active_trainset.features, passive_trainset.features), axis=1)
    x_test = np.concatenate((active_testset.features, passive_testset.features), axis=1)
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

