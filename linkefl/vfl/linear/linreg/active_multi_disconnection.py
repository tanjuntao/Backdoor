import copy
import time

import numpy as np
from sklearn.metrics import r2_score
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory_multi_disconnection
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import add_intercept, AddIntercept
from  linkefl.vfl.linear.linreg.multi_disconnection import ActiveLinReg_disconnection



if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'diabetes'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = ['localhost', 'localhost']
    active_port = [20000, 30000]
    passive_ip = ['localhost', 'localhost']
    passive_port = [20001, 30001]
    world_size = 2
    _epochs = 200000
    _batch_size = -1
    _learning_rate = 1.0
    _penalty = Const.NONE
    _reg_lambda = 0.01
    _crypto_type = Const.PLAIN
    _random_state = None
    _key_size = 1024
    _val_freq = 5000

    reconnection = False
    reconnection_port = [20002, 30002]

    # 1. Loading dataset and preprocessing
    # Option 1: Scikit-learn style
    print('Loading dataset...')
    active_trainset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                   dataset_name=dataset_name,
                                                   root='../../data',
                                                   train=True,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    active_testset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                  dataset_name=dataset_name,
                                                  root='../../data',
                                                  train=False,
                                                  download=True,
                                                  passive_feat_frac=passive_feat_frac,
                                                  feat_perm_option=feat_perm_option)
    active_trainset = add_intercept(active_trainset)
    active_testset = add_intercept(active_testset)
    print('Done.')

    # Option 2: PyTorch style
    print('Loading dataset...')
    # transform = AddIntercept()
    # active_trainset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
    #                                                dataset_name=dataset_name,
    #                                                train=True,
    #                                                passive_feat_frac=passive_feat_frac,
    #                                                feat_perm_option=feat_perm_option,
    #                                                transform=transform)
    # active_testset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
    #                                               dataset_name=dataset_name,
    #                                               train=False,
    #                                               passive_feat_frac=passive_feat_frac,
    #                                               feat_perm_option=feat_perm_option,
    #                                               transform=transform)
    # print('Done.')

    # 3. Initialize cryptosystem
    _crypto = crypto_factory(crypto_type=_crypto_type,
                             key_size=_key_size,
                             num_enc_zeros=10000,
                             gen_from_set=False)

    # 4. Initialize messenger
    _messenger = messenger_factory_multi_disconnection(messenger_type=Const.FAST_SOCKET,
                                   role=Const.ACTIVE_NAME,
                                   model_type="NN",
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port,
                                   world_size=world_size)

    print('ACTIVE PARTY started, connecting...')

    # 5. Initialize model and start training
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    active_party = ActiveLinReg_disconnection(epochs=_epochs,
                                batch_size=_batch_size,
                                learning_rate=_learning_rate,
                                messenger=_messenger,
                                cryptosystem=_crypto,
                                logger=_logger,
                                penalty=_penalty,
                                reg_lambda=_reg_lambda,
                                random_state=_random_state,
                                val_freq=_val_freq,
                                saving_model=False,
                                world_size=world_size,
                                reconnection=reconnection,
                                reconnection_port=reconnection_port
                              )

    active_party.train(active_trainset, active_testset)

    # 6. Close messenger, finish training.
    for msger_ in _messenger:
        msger_.close()

    # # For online inference, you only need to substitue the model name
    # scores = ActiveLinReg.online_inference(
    #     active_testset,
    #     model_name='20220831_190241-active_party-vertical_linreg-402_samples.model',
    #     messenger=_messenger
    # )
    # print(scores)
