import os.path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from linkefl.dataio import NumpyDataset
from linkefl.feature.transform.functional import bin


class FeatureEvaluation(object):
    def __init__(self):
        pass

    @classmethod
    def tree_importance(
        cls,
        trainset: NumpyDataset,
        task: str = "binary",
        evaluation_way: str = "xgboost",
        importance_type: str = "gain",
        save_pic: bool = True,
        max_num_features_plot: int = 25,
        pic_path: str = "./eval_results",
    ):
        """Measure feature importance based on tree model."""

        # 0. param check
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"

        # 1. set model and train
        if task == "regression":
            model = XGBRegressor(eval_metric="rmse")
            model.fit(X=trainset.features, y=trainset.labels, verbose=False)
        elif task == "binary":
            model = XGBClassifier(eval_metric="logloss")
            model.fit(X=trainset.features, y=trainset.labels, verbose=False)
        elif task == "multi":
            model = XGBClassifier(eval_metric="mlogloss")
            model.fit(X=trainset.features, y=trainset.labels, verbose=False)
        else:
            raise ValueError("Unsupported task class.")

        # 2. feature evaluation
        if evaluation_way == "xgboost":
            importances = model.feature_importances_
            ranking = np.argsort(importances)[::-1]

            if save_pic:
                xgb.plot_importance(
                    model,
                    importance_type=importance_type,
                    max_num_features=max_num_features_plot,
                )  # 'weight', 'gain', 'cover'
                plt.xlabel(f"{importance_type}", labelpad=5, loc="center")
                plt.savefig(
                    os.path.join(pic_path, "feature_importance_gain.png"),
                    bbox_inches="tight",
                )
                plt.close()
                print("save importance fig success.")

        elif evaluation_way == "shap":
            explainer = shap.Explainer(model)
            shap_values = explainer(trainset.features)
            if task == "regression" or task == "binary":
                # in binary case, shape of shap_values is (n_samples, n_features)
                importances = np.mean(
                    np.abs(shap_values.values), axis=0
                )  # summation along the vertical axis
                ranking = np.argsort(importances)[::-1]
            elif task == "multi":
                # in multiclass case, shape of shap_values is
                # (n_samples, n_features, n_classes)

                # first compute mean along the n_classes axis, then compute mean
                # along the n_samples axis
                importances = np.mean(
                    np.mean(np.abs(shap_values.values), axis=2), axis=0
                )
                ranking = np.argsort(importances)[::-1]
        else:
            raise ValueError("Unsupported evaluation way.")

        return importances, ranking

    @classmethod
    def collinearity_anay(
        cls,
        dateset: NumpyDataset,
        evaluation_way: str = "pearson",
        save_pic: bool = True,
        max_num_features_plot: int = 5,
        pic_path: str = "./eval_results",
    ):
        """Using Variance Inflation Factor for characteristic collinearity analysis."""
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

        if save_pic:
            corr = np.array(corr)
            bound = (
                max_num_features_plot
                if corr.shape[1] > max_num_features_plot
                else corr.shape[1]
            )
            corr_plot = corr[:bound, :bound]
            sns.heatmap(
                corr_plot,
                linewidths=0.1,
                vmax=1.0,
                square=True,
                linecolor="white",
                annot=True,
            )
            plt.savefig(os.path.join(pic_path, "collinearity_anay.png"))
            plt.close()
            print("save collinearity_anay.png success.")

        return corr

    @classmethod
    def calculate_psi(
        cls,
        trainset,
        testset,
        save_pic: bool = True,
        max_num_features_plot: int = 10,
        pic_path: str = "./eval_results",
    ):
        psi_all = []
        for i in range(trainset.features.shape[1]):
            psi, _ = cls._calculate_feature_psi(
                trainset.features[:, i], testset.features[:, i]
            )
            psi_all.append(psi)

        if save_pic:
            features = [f"f{i}" for i in range(len(psi_all))]
            tuples = sorted(zip(features, psi_all), key=lambda x: x[1])
            tuples = tuples[:max_num_features_plot]
            features_plot, values_plot = zip(*tuples)

            cls._plot_bar(features_plot, values_plot, pic_path=pic_path)
            plt.close()

        return psi_all

    @classmethod
    def _calculate_feature_psi(cls, base_list, test_list, bins=20, min_sample=10):
        try:
            base_df = pd.DataFrame(base_list, columns=["score"])
            test_df = pd.DataFrame(test_list, columns=["score"])

            # 1.去除缺失值后，统计两个分布的样本量
            base_notnull_cnt = len(list(base_df["score"].dropna()))
            test_notnull_cnt = len(list(test_df["score"].dropna()))

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
                    bk = base_df["score"].quantile(q)
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
                bucket_list.append("(" + str(left) + "," + str(right) + "]")

                base_cnt = base_df[
                    (base_df.score > left) & (base_df.score <= right)
                ].shape[0]
                base_cnt_list.append(base_cnt)

                test_cnt = test_df[
                    (test_df.score > left) & (test_df.score <= right)
                ].shape[0]
                test_cnt_list.append(test_cnt)

            # 5.汇总统计结果
            stat_df = pd.DataFrame(
                {
                    "bucket": bucket_list,
                    "base_cnt": base_cnt_list,
                    "test_cnt": test_cnt_list,
                }
            )
            stat_df["base_dist"] = stat_df["base_cnt"] / len(base_df)
            stat_df["test_dist"] = stat_df["test_cnt"] / len(test_df)

            def sub_psi(row):
                # 6.计算PSI
                base_list = row["base_dist"]
                test_dist = row["test_dist"]
                # 处理某分箱内样本量为0的情况
                if base_list == 0 and test_dist == 0:
                    return 0
                elif base_list == 0 and test_dist > 0:
                    base_list = 1 / base_notnull_cnt
                elif base_list > 0 and test_dist == 0:
                    test_dist = 1 / test_notnull_cnt

                return (test_dist - base_list) * np.log(test_dist / base_list)

            stat_df["psi"] = stat_df.apply(lambda row: sub_psi(row), axis=1)
            stat_df = stat_df[
                ["bucket", "base_cnt", "base_dist", "test_cnt", "test_dist", "psi"]
            ]
            psi = stat_df["psi"].sum()

        except Exception:
            print("error!!!")
            psi = np.nan
            stat_df = None

        return psi, stat_df

    @classmethod
    def _plot_bar(
        cls,
        features: list,
        values: list,
        title: str = "psi anay",
        xlabel: str = "psi score",
        ylabel: str = "Features",
        # figsize: Optional[Tuple[float, float]] = None, # raise Cythoning error
        figsize: Optional[tuple] = (10, 6),
        height: float = 0.2,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        grid: bool = True,
        show_values: bool = True,
        precision: Optional[int] = 5,
        pic_path: str = "./eval_results",
    ):
        # set ax
        if figsize is not None:
            cls._check_not_tuple_of_2_elements(figsize, "figsize")
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ylocs = np.arange(len(values))
        ax.barh(ylocs, values, align="center", height=height)

        gap = min(1, max(values) * 0.02)  # avoid errors when the value is less than 1
        if show_values is True:
            for x, y in zip(values, ylocs):
                ax.text(x + gap, y, cls._float2str(x, precision), va="center")

        ax.set_yticks(ylocs)
        ax.set_yticklabels(features)

        # Set the x-axis scope
        if xlim is not None:
            cls._check_not_tuple_of_2_elements(xlim, "xlim")
        else:
            xlim = (0, max(values) * 1.1)
        ax.set_xlim(xlim)

        # Set the y-axis scope
        if ylim is not None:
            cls._check_not_tuple_of_2_elements(ylim, "ylim")
        else:
            ylim = (-1, len(values))
        ax.set_ylim(ylim)

        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.grid(grid)

        plt.savefig(os.path.join(pic_path, "psi_anay.png"), pad_inches="tight")

        return ax

    @staticmethod
    def _check_not_tuple_of_2_elements(obj: Any, obj_name: str = "obj") -> None:
        """Check object is not tuple or does not have 2 elements."""
        if not isinstance(obj, tuple) or len(obj) != 2:
            raise TypeError(f"{obj_name} must be a tuple of 2 elements.")

    @staticmethod
    def _float2str(value: float, precision: Optional[int] = None) -> str:
        return (
            f"{value:.{precision}f}"
            if precision is not None and not isinstance(value, str)
            else str(value)
        )


if __name__ == "__main__":
    from linkefl.common.const import Const
    from linkefl.dataio import NumpyDataset

    np_dataset = NumpyDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path="/Users/tanjuntao/LinkeFL-Servicer/data/电商平台精准营销数据202206.csv",
        dataset_type=Const.CLASSIFICATION,
        has_header=False,
    )
    print(np_dataset.features.shape)
    # FeatureEvaluation.tree_importance(np_dataset, pic_path="./")
    # FeatureEvaluation.collinearity_anay(np_dataset, pic_path="./")
    # trainset, testset = NumpyDataset.train_test_split(np_dataset, test_size=0.2)
    # FeatureEvaluation.calculate_psi(trainset, testset, pic_path="./")

    np_dataset01 = NumpyDataset.feature_split(np_dataset, n_splits=10)[1]
    print(np_dataset01.features.shape, np_dataset01.header)
    np_dataset01.describe()
