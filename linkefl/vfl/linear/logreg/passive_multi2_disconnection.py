import numpy as np
import copy
import time

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory, messenger_factory,messenger_factory_multi_disconnection
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale, Scale
# from linkefl.vfl.linear import BaseLinearPassive
from  linkefl.vfl.linear.base_multi import BaseLinearPassive
from linkefl.modelio import NumpyModelIO
from linkefl.common.factory import (
    crypto_factory,
    messenger_factory,
    partial_crypto_factory,
)
from linkefl.vfl.linear.logreg.multi_disconnection import PassiveLogReg_disconnection





if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'census'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 30000
    passive_ip = 'localhost'
    passive_port = 30001
    _epochs = 100
    _batch_size = 100
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.01
    _random_state = 3347
    _crypto_type = Const.PLAIN
    _using_pool = False

    # 1. Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print('Loading dataset...')
    passive_trainset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
                                                    dataset_name=dataset_name,
                                                    root='../../data',
                                                    train=True,
                                                    download=True,
                                                    passive_feat_frac=passive_feat_frac,
                                                    feat_perm_option=feat_perm_option)
    passive_testset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
                                                   dataset_name=dataset_name,
                                                   root='../../data',
                                                   train=False,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    passive_trainset = NumpyDataset.feature_split(passive_trainset, n_splits=2)[1]
    passive_testset = NumpyDataset.feature_split(passive_testset, n_splits=2)[1]
    # load dummy dataset
    # dummy_dataset = NumpyDataset.dummy_daaset(
    #     role=Const.PASSIVE_NAME,
    #     dataset_type=Const.CLASSIFICATION,
    #     n_samples=100000,
    #     n_features=100,
    #     passive_feat_frac=passive_feat_frac
    # )
    # passive_trainset, passive_testset = NumpyDataset.train_test_split(dummy_dataset, test_size=0.2)
    passive_trainset = scale(passive_trainset)
    passive_testset = scale(passive_testset)

    # Option 2: PyTorch style
    # print('Loading dataset...')
    # transform = Scale()
    # passive_trainset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
    #                                                 dataset_name=dataset_name,
    #                                                 train=True,
    #                                                 passive_feat_frac=passive_feat_frac,
    #                                                 feat_perm_option=feat_perm_option,
    #                                                 transform=transform)
    # passive_testset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
    #                                                dataset_name=dataset_name,
    #                                                train=False,
    #                                                passive_feat_frac=passive_feat_frac,
    #                                                feat_perm_option=feat_perm_option,
    #                                                transform=transform)
    # print('Done.')

    # 3. Initialize messenger
    _messenger = messenger_factory_multi_disconnection(messenger_type=Const.FAST_SOCKET,
                                  role=Const.PASSIVE_NAME,
                                  model_type="NN",
                                  active_ip=active_ip,
                                  active_port=active_port,
                                  passive_ip=passive_ip,
                                  passive_port=passive_port)

    # 4. Initialize model and start training
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    passive_party = PassiveLogReg_disconnection(epochs=_epochs,
                                  batch_size=_batch_size,
                                  learning_rate=_learning_rate,
                                  messenger=_messenger,
                                  crypto_type=_crypto_type,
                                  logger=_logger,
                                  penalty=_penalty,
                                  reg_lambda=_reg_lambda,
                                  random_state=_random_state,
                                  using_pool=_using_pool,
                                  saving_model=False,)

    passive_party.train(passive_trainset, passive_testset)

    # 5. Close messenger, finish training
    _messenger.close()

    # For online inference, you just need to substitute the model_name
    # scores = PassiveLogReg.online_inference(
    #     passive_testset,
    #     model_name='20220831_185109-passive_party-vertical_logreg-455_samples.model',
    #     messenger=_messenger
    # )
    # print(scores)
