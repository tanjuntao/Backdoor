import numpy as np
import pandas as pd
from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import parse_label
from linkefl.feature.feature_evaluation import FeatureEvaluation
from linkefl.feature.woe import ActiveWoe
from linkefl.feature.chi_square_bin import ActiveChiBin
from linkefl.feature.pearson import ActivePearsonVfl


if __name__ == "__main__":
    # 0. Set parameters
    #  binary: cancer, digits, epsilon, census, credit, default_credit, criteo
    #  regression: diabetes
    dataset_name = "census"
    passive_feat_frac = 0.1
    feat_perm_option = Const.SEQUENCE
    crypto_type = Const.FAST_PAILLIER
    key_size = 1024
    active_ip = ['localhost', ]
    active_port = [20000, ]
    passive_ip = ['localhost', ]
    passive_port = [20002, ]

    # 1. Load datasets
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                   dataset_name=dataset_name,
                                                   root='../vfl/data',
                                                   train=True,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    active_testset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                  dataset_name=dataset_name,
                                                  root='../vfl/data',
                                                  train=False,
                                                  download=True,
                                                  passive_feat_frac=passive_feat_frac,
                                                  feat_perm_option=feat_perm_option)
    active_trainset = parse_label(active_trainset)
    active_testset = parse_label(active_testset)
    print("Done")
    # 2. Initialize cryptosystem
    _crypto = crypto_factory(crypto_type=crypto_type,
                             key_size=key_size,
                             num_enc_zeros=10,
                             gen_from_set=False)

    # 3. Initialize messenger
    _messenger = [
        messenger_factory(messenger_type=Const.FAST_SOCKET,
                          role=Const.ACTIVE_NAME,
                          active_ip=ac_ip,
                          active_port=ac_port,
                          passive_ip=pass_ip,
                          passive_port=pass_port,
                          )
        for ac_ip, ac_port, pass_ip, pass_port in
        zip(active_ip, active_port, passive_ip, passive_port)
    ]

    active_trainset = parse_label(active_trainset)
    active_testset = parse_label(active_testset)
    print("Done")

    importances, ranking = FeatureEvaluation.tree_importance(active_trainset, task="binary",evaluation_way="xgboost")
    print(importances, ranking)

    corr = FeatureEvaluation.collinearity_anay(dateset=active_trainset, evaluation_way="pearson")
    print(corr)
    
    feature_psi = FeatureEvaluation.calculate_psi(active_trainset, active_testset)
    print(feature_psi)

    # split, woe, iv = ActiveWoe(dataset=active_trainset, idxes=[2, 3], messenger=_messenger).cal_woe()
    # print(split, woe, iv)

    # chi_bin = ActiveChiBin(dataset=active_trainset, idxes=[2, 3], messenger=_messenger, max_group=200).chi_bin()
    # print(chi_bin)

    # ActivePearsonVfl(dataset=active_trainset, messenger=_messenger, cryptosystem=_crypto).pearson_vfl()

    # pearson = ActivePearsonVfl(dataset=active_trainset, messenger=_messenger, cryptosystem=_crypto).pearson_single()
    # print(pearson)
