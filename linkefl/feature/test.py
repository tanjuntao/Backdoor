import numpy as np
import pandas as pd
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import parse_label
from linkefl.feature.feature_evaluation import FeatureEvaluation

if __name__ == "__main__":
    # 0. Set parameters
    #  binary: cancer, digits, epsilon, census, credit, default_credit, criteo
    #  regression: diabetes
    dataset_name = "census"
    passive_feat_frac = 0.1
    feat_perm_option = Const.SEQUENCE

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

    importances, ranking = FeatureEvaluation.tree_importance(active_trainset, active_testset, task="binary",evaluation_way="xgboost")
    print(importances, ranking)

    # corr = FeatureEvaluation.collinearity_anay(dateset=active_trainset, evaluation_way="pearson")
    # print(corr)
    #
    # feature_psi = FeatureEvaluation.calculate_psi(active_trainset, active_testset)
    # print(feature_psi)
