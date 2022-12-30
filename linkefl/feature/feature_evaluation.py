import shap
import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor

from linkefl.dataio import NumpyDataset
from linkefl.feature.transform.functional import bin

class FeatureEvaluation(object):
    def __init__(self):
        pass

    @classmethod
    def tree_importance(cls,
                        trainset: NumpyDataset, testset: NumpyDataset,
                        task: str="binary",
                        evaluation_way: str="xgboost"):
        """Measure feature importance based on tree model.
        """

        # 0. param check
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        # 1. set model and train
        if task == "regression":
            model = XGBRegressor(eval_metric="rmse")
            model.fit(X=trainset.features, y=trainset.labels,
                      verbose=False)
        elif task == "binary":
            model = XGBClassifier(eval_metric="logloss")
            model.fit(X=trainset.features, y=trainset.labels,
                      verbose=False)
        elif task == "multi":
            model = XGBClassifier(eval_metric="mlogloss")
            model.fit(X=trainset.features, y=trainset.labels,
                      verbose=False)
        else:
            raise ValueError("Unsupported task class.")

        # 2. feature evaluation
        if evaluation_way == 'xgboost':
            importances = model.feature_importances_
            ranking = np.argsort(importances)[::-1]
        elif evaluation_way == 'shap':
            explainer = shap.Explainer(model)
            shap_values = explainer(trainset.features)
            if task == "regression" or task == "binary":
                # in binary case, shape of shap_values is (n_samples, n_features)
                importances = np.mean(np.abs(shap_values.values),
                                      axis=0)  # summation along the vertical axis
                ranking = np.argsort(importances)[::-1]
            elif task == "multi":
                # in multiclass case, shape of shap_values is
                # (n_samples, n_features, n_classes)

                # first compute mean along the n_classes axis, then compute mean
                # along the n_samples axis
                importances = np.mean(np.mean(np.abs(shap_values.values), axis=2),
                                      axis=0)
                ranking = np.argsort(importances)[::-1]
        else:
            raise ValueError("Unsupported evaluation way.")

        return importances, ranking

    @classmethod
    def collinearity_anay(cls, dateset: NumpyDataset, evaluation_way: str="pearson"):
        """Using Variance Inflation Factor for characteristic collinearity analysis.
        """
        # 0. param check
        assert isinstance(
            dateset, NumpyDataset
        ), "dataset should be an instance of NumpyDataset"

        # 1. collinearity_anay
        if evaluation_way == "pearson" or evaluation_way == "spearman":
            features = pd.DataFrame(dateset.features)
            corr = features.corr(method=f"{evaluation_way}")
        elif evaluation_way == "vif":
            # todo: to be write.
            raise NotImplementedError("to be done.")
        else:
            raise ValueError("Unsupported evaluation way")

        return corr

    @classmethod
    def calculate_psi(cls, trainset, testset):
        psi_all = []
        for i in range(trainset.features.shape[1]):
            psi, _ = cls._calculate_feature_psi(trainset.features[:, i], testset.features[:, i])
            psi_all.append(psi)

        return psi_all

    @classmethod
    def _calculate_feature_psi(cls, base_list, test_list, bins=20, min_sample=10):
        try:
            base_df = pd.DataFrame(base_list, columns=['score'])
            test_df = pd.DataFrame(test_list, columns=['score'])

            # 1.去除缺失值后，统计两个分布的样本量
            base_notnull_cnt = len(list(base_df['score'].dropna()))
            test_notnull_cnt = len(list(test_df['score'].dropna()))

            # 空分箱
            base_null_cnt = len(base_df) - base_notnull_cnt
            test_null_cnt = len(test_df) - test_notnull_cnt

            # 2.最小分箱数
            q_list = []
            if type(bins) == int:
                bin_num = min(bins, int(base_notnull_cnt / min_sample))
                q_list = [x / bin_num for x in range(1, bin_num)]
                break_list = []
                for q in q_list:
                    bk = base_df['score'].quantile(q)
                    break_list.append(bk)
                break_list = sorted(list(set(break_list)))  # 去重复后排序
                score_bin_list = [-np.inf] + break_list + [np.inf]
            else:
                score_bin_list = bins

            # 4.统计各分箱内的样本量
            base_cnt_list = [base_null_cnt]
            test_cnt_list = [test_null_cnt]
            bucket_list = ["MISSING"]
            for i in range(len(score_bin_list) - 1):
                left = round(score_bin_list[i + 0], 4)
                right = round(score_bin_list[i + 1], 4)
                bucket_list.append("(" + str(left) + ',' + str(right) + ']')

                base_cnt = base_df[(base_df.score > left) & (base_df.score <= right)].shape[0]
                base_cnt_list.append(base_cnt)

                test_cnt = test_df[(test_df.score > left) & (test_df.score <= right)].shape[0]
                test_cnt_list.append(test_cnt)

            # 5.汇总统计结果
            stat_df = pd.DataFrame({"bucket": bucket_list, "base_cnt": base_cnt_list, "test_cnt": test_cnt_list})
            stat_df['base_dist'] = stat_df['base_cnt'] / len(base_df)
            stat_df['test_dist'] = stat_df['test_cnt'] / len(test_df)

            def sub_psi(row):
                # 6.计算PSI
                base_list = row['base_dist']
                test_dist = row['test_dist']
                # 处理某分箱内样本量为0的情况
                if base_list == 0 and test_dist == 0:
                    return 0
                elif base_list == 0 and test_dist > 0:
                    base_list = 1 / base_notnull_cnt
                elif base_list > 0 and test_dist == 0:
                    test_dist = 1 / test_notnull_cnt

                return (test_dist - base_list) * np.log(test_dist / base_list)

            stat_df['psi'] = stat_df.apply(lambda row: sub_psi(row), axis=1)
            stat_df = stat_df[['bucket', 'base_cnt', 'base_dist', 'test_cnt', 'test_dist', 'psi']]
            psi = stat_df['psi'].sum()

        except:
            print('error!!!')
            psi = np.nan
            stat_df = None

        return psi, stat_df

