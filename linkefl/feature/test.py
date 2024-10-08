import numpy as np
import pandas as pd

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.dataio import NumpyDataset, TorchDataset
from linkefl.feature.chi_square_bin import ActiveChiBin
from linkefl.feature.feature_evaluation import FeatureEvaluation
from linkefl.feature.pearson import ActivePearson
from linkefl.feature.transform import parse_label
from linkefl.feature.woe import ActiveWoe

if __name__ == "__main__":
    # 0. Set parameters
    #  binary: cancer, digits, epsilon, census, credit, default_credit, criteo
    #  regression: diabetes
    dataset_name = "census"
    passive_feat_frac = 0.1
    feat_perm_option = Const.SEQUENCE
    crypto_type = Const.FAST_PAILLIER
    key_size = 1024
    active_ip = [
        "localhost",
    ]
    active_port = [
        20001,
    ]
    passive_ip = [
        "localhost",
    ]
    passive_port = [
        20003,
    ]

    # 1. Load datasets
    print("Loading dataset...")
    # active_trainset = NumpyDataset.buildin_dataset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_name=dataset_name,
    #     root='../vfl/data',
    #     train=True,
    #     download=True,
    #     passive_feat_frac=passive_feat_frac,
    #     feat_perm_option=feat_perm_option
    # )
    # active_testset = NumpyDataset.buildin_dataset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_name=dataset_name,
    #     root='../vfl/data',
    #     train=False,
    #     download=True,
    #     passive_feat_frac=passive_feat_frac,
    #     feat_perm_option=feat_perm_option
    # )
    active_trainset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.ACTIVE_NAME,
        root="../vfl/data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_testset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.ACTIVE_NAME,
        root="../vfl/data",
        train=False,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    # active_trainset = parse_label(active_trainset)
    # active_testset = parse_label(active_testset)
    print("Done")
    # 2. Initialize cryptosystem
    _crypto = crypto_factory(
        crypto_type=crypto_type, key_size=key_size, num_enc_zeros=10, gen_from_set=False
    )

    # 3. Initialize messenger
    _messengers = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=ac_ip,
            active_port=ac_port,
            passive_ip=pass_ip,
            passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in zip(
            active_ip, active_port, passive_ip, passive_port
        )
    ]

    # active_trainset = parse_label(active_trainset)
    # active_testset = parse_label(active_testset)
    print("Done")

    # importances, ranking = FeatureEvaluation.tree_importance(
    #     active_trainset,
    #     task="binary",
    #     evaluation_way="xgboost"
    # )
    # print(importances, ranking)

    # corr = FeatureEvaluation.collinearity_anay(
    #     dateset=active_trainset,
    #     evaluation_way="pearson"
    # )
    # print(corr)

    # feature_psi = FeatureEvaluation.calculate_psi(active_trainset, active_testset)
    # print(feature_psi)

    # 明文计算主动方本地的woe & iv值
    cal_woe = ActiveWoe(
        idxes=[2, 3],
        modify=False,
        crypto_type=Const.PLAIN,
        messengers=_messengers
    )
    cal_woe(active_trainset, Const.ACTIVE_NAME)
    woe, iv = cal_woe.bin_woe, cal_woe.bin_iv
    print(woe, iv)
    # 密文计算被动方的woe & iv值
    # cal_woe = ActiveWoe(
    #     idxes=[2, 3],
    #     modify=False,
    #     crypto_type=Const.PAILLIER,
    #     messengers=_messengers,
    #     cryptosystem=_crypto,
    # )
    # cal_woe(dataset=active_trainset, role=Const.ACTIVE_NAME)
    # woe, iv = cal_woe.bin_woe, cal_woe.bin_iv
    # print(woe, iv)

    # chi_bin = ActiveChiBin(
    #     dataset=active_trainset,
    #     idxes=[2, 3],
    #     messenger=_messenger,
    #     max_group=200
    # ).chi_bin()
    # print(chi_bin)

    # ActivePearson(
    #     dataset=active_trainset,
    #     messenger=_messenger,
    #     cryptosystem=_crypto
    # ).pearson_vfl()

    # pearson = ActivePearson(
    #     dataset=active_trainset,
    #     messenger=_messenger,
    #     cryptosystem=_crypto
    # ).pearson_single()
    # print(pearson)
