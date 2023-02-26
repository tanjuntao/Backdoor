#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LinkeFL 
# @File    : evaluate.py.py
# @Author  : HaoRan Cheng
# @Date    : 2023/2/25 21:09

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd

from typing import Any, Optional
from sklearn.metrics import precision_recall_curve, roc_curve

Axes = Any  # real type is matplotlib.axes.Axes
GraphvizSource = Any  # real type is graphviz.Source
ActiveTreeParty = Any  # real type is linkefl.vfl.tree.ActiveTreeParty


class Evaluate(object):
    def __init__(self):
        pass

    @staticmethod
    def eval_ks(labels, y_probs, cut_point=500):
        good_len = len([x for x in labels if x == 0])
        bad_len = len([x for x in labels if x == 1])
        pred_prob_labels = list(zip(y_probs, labels))

        good_point = []
        bad_point = []
        diff_point = []  # 记录每个阈值点下的KS值
        thresholds = np.linspace(0, 1, cut_point)
        for i in thresholds:
            hit_data = [x[1] for x in pred_prob_labels if x[0] > i]  # 选取当前阈值下的数据
            good_hit = len([x for x in hit_data if x == 0])  # 当前阈值下真实为0但预测为1 FP
            bad_hit = len([x for x in hit_data if x == 1])  # 当前阈值下真实为1且预测为1 TP
            good_rate = good_hit / good_len  # FPR
            bad_rate = bad_hit / bad_len  # TPR
            diff = abs(good_rate - bad_rate)  # KS值
            good_point.append(good_rate)
            bad_point.append(bad_rate)
            diff_point.append(diff)

        ks_value = max(diff_point)  # 获得最大KS值为KS值
        ks_x_axis = diff_point.index(ks_value)  # KS值下的阈值点索引
        threshold = thresholds[ks_x_axis]  # 阈值

        return ks_value, threshold


class TreePrint(object):
    def __int__(self):
        pass

    @classmethod
    def _prepare_print_val(cls, tree, root):
        if not root:
            return

        if root.value is not None:
            # leaf node
            print_val = f"value: {root.value: .3f}"
        else:
            # mid node
            if root.party_id == 0:
                print_val = "active_party\n"
                print_val += f"record_id: {root.record_id}\n"
                print_val += f"feature: f{int(tree.record[root.record_id][0])}\n"
                print_val += f"threshold: {tree.record[root.record_id][1]: .3f}"
            else:
                print_val = f"passive_party_{root.party_id}\n"
                print_val += f"record_id: {root.record_id}\n"
                print_val += "feature: encrypt\n"
                print_val += "threshold: encrypt"

        root.print_val = print_val
        root.children = []
        # print(root.print_val)
        if root.left_branch:
            cls._prepare_print_val(tree, root.left_branch)
            root.children.append(root.left_branch)
        if root.right_branch:
            cls._prepare_print_val(tree, root.right_branch)
            root.children.append(root.right_branch)

    @classmethod
    def tree_to_str(cls, tree, tree_structure):
        """

        Args:
            tree: DecisionTree
            tree_structure: str, can be "HORIZONTAL" or "VERTICAL"

        Returns:
            str for tree structure.
        """
        try:
            from PrettyPrint import PrettyPrintTree
        except ImportError as e:
            raise ImportError("You must install PrettyPrint to plot tree") from e

        root = tree.root
        cls._prepare_print_val(tree, root)

        orientation = (
            PrettyPrintTree.HORIZONTAL
            if tree_structure == "HORIZONTAL"
            else PrettyPrintTree.VERTICAL
        )

        pt = PrettyPrintTree(
            get_children=lambda x: x.children if x else [],
            get_val=lambda x: x.print_val if x else "",
            default_orientation=orientation,
            border=True,
            color=None,
            return_instead_of_print=True,
        )

        tree_str = pt(root)
        return tree_str


class Plot(object):
    def __init__(self):
        pass

    @staticmethod
    def plot_train_test_loss(train_loss, test_loss, file_dir="./models"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            list(range(len(train_loss))), train_loss, label="train_loss"
        )  # color='darkorange'
        ax.plot(list(range(len(test_loss))), test_loss, label="test_loss")
        # ax.set_xlim(0, len(test_loss)-0.95)
        # ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.grid(True, linestyle="-.")
        ax.set_title("Convergence Analysis")
        ax.set_ylabel("loss", labelpad=5, loc="center")
        ax.set_xlabel("epoch", labelpad=5, loc="center")
        plt.legend(loc="best")

        plt.savefig(f"{file_dir}/convergence_analysis_loss.png")
        plt.close()

    @staticmethod
    def plot_train_test_auc(train_auc, test_auc, file_dir="./models"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            list(range(len(train_auc))), train_auc, label="train_auc"
        )  # color='darkorange'
        ax.plot(list(range(len(test_auc))), test_auc, label="test_auc")
        # ax.set_xlim(0, len(train_auc) - 0.95)
        # ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.grid(True, linestyle="-.")
        ax.set_title("Convergence Index Analysis")
        ax.set_ylabel("Auc", labelpad=5, loc="center")
        ax.set_xlabel("Epoch", labelpad=5, loc="center")
        plt.legend(loc="best")

        plt.savefig(f"{file_dir}/convergence_index_analysis_auc.png")
        plt.close()

    @classmethod
    def plot_binary_mertics(cls, labels, y_probs, cut_point=500, file_dir: str = "./models"):
        cls._plot_pr(labels, y_probs, file_dir)
        cls._plot_roc(labels, y_probs, file_dir)
        cls._plot_ks(labels, y_probs, cut_point=cut_point, file_dir=file_dir)
        cls._plot_lift(labels, y_probs, file_dir)

    @staticmethod
    def plot_regression_metrics(mae, mse, sse, r2, file_dir: str = "./models"):
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.plot(list(range(len(mae))), mae, ls="-", linewidth=2.0)
        ax.grid(True, linestyle="-.")
        ax.set_xlabel("epoch", labelpad=5, loc="center")
        ax.set_ylabel("mae", labelpad=5, loc="center")
        ax.set_title("MAE Curve")

        ax = fig.add_subplot(2, 2, 2)
        ax.plot(list(range(len(mse))), mse, ls="-", linewidth=2.0)
        ax.grid(True, linestyle="-.")
        ax.set_xlabel("epoch", labelpad=5, loc="center")
        ax.set_ylabel("mse", labelpad=5, loc="center")
        ax.set_title("MSE Curve")

        ax = fig.add_subplot(2, 2, 3)
        ax.plot(list(range(len(sse))), sse, ls="-", linewidth=2.0)
        ax.grid(True, linestyle="-.")
        ax.set_xlabel("epoch", labelpad=5, loc="center")
        ax.set_ylabel("sse", labelpad=5, loc="center")
        ax.set_title("SSE Curve")

        ax = fig.add_subplot(2, 2, 4)
        ax.plot(list(range(len(r2))), r2, ls="-", linewidth=2.0)
        ax.grid(True, linestyle="-.")
        ax.set_xlabel("epoch", labelpad=5, loc="center")
        ax.set_ylabel("r2", labelpad=5, loc="center")
        ax.set_title("R2 Curve")

        plt.tight_layout()
        plt.savefig(f"{file_dir}/regression_metric.png")
        # plt.show()
        plt.close()

    @classmethod
    def _plot_pr(cls, label, y_prob, file_dir):
        precision, recall, thresholds_pr = precision_recall_curve(label, y_prob)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(recall, precision, ls="-", linewidth=2.0)
        ax.grid(True, linestyle="-.")
        ax.set_xlabel("recall", labelpad=5, loc="center")
        ax.set_ylabel("precision", labelpad=5, loc="center")
        ax.set_title("PR Curve")

        plt.savefig(f"{file_dir}/PR_Curve.png")
        plt.close()

    @classmethod
    def _plot_roc(cls, label, y_prob, file_dir):
        fpr, tpr, thresholds_roc = roc_curve(label, y_prob)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(fpr, tpr, ls="-", linewidth=2.0)
        ax.grid(True, linestyle="-.")
        ax.set_xlabel("false positive rate", labelpad=5, loc="center")
        ax.set_ylabel("true positive rate", labelpad=5, loc="center")
        ax.set_title("ROC Curve")

        plt.savefig(f"{file_dir}/ROC_Curve.png")
        plt.close()

    @classmethod
    def _plot_ks(cls, labels, y_probs, cut_point=500, file_dir="./models"):
        good_len = len([x for x in labels if x == 0])  # y=0数量
        bad_len = len([x for x in labels if x == 1])  # y=1数量
        pred_prob_labels = list(zip(y_probs, labels))

        good_point = []
        bad_point = []
        diff_point = []  # 记录每个阈值点下的KS值
        thresholds = np.linspace(0, 1, cut_point)

        for i in thresholds:
            hit_data = [x[1] for x in pred_prob_labels if x[0] > i]  # 选取当前阈值下的数据
            good_hit = len([x for x in hit_data if x == 0])  # 当前阈值下真实为0但预测为1 FP
            bad_hit = len([x for x in hit_data if x == 1])  # 当前阈值下真实为1且预测为1 TP
            good_rate = good_hit / good_len  # FPR
            bad_rate = bad_hit / bad_len  # TPR
            diff = abs(good_rate - bad_rate)  # KS值
            good_point.append(good_rate)
            bad_point.append(bad_rate)
            diff_point.append(diff)

        ks_value = max(diff_point)  # 获得最大KS值为KS值
        ks_x_axis = diff_point.index(ks_value)  # KS值下的阈值点索引
        ks_good_point, ks_bad_point = good_point[ks_x_axis], bad_point[ks_x_axis]  # 阈值下y=0,y=1在组内的占比
        threshold = thresholds[ks_x_axis]  # 阈值

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True, linestyle="-.")

        ax.plot(thresholds, good_point, color="green", label="FPR", linewidth=2)
        ax.plot(thresholds, bad_point, color="red", label="TPR", linewidth=2)
        ax.plot(thresholds, diff_point, color="darkorange", alpha=0.5, label="KS", linewidth=2)
        ax.plot([threshold, threshold], [0, 1], linestyle="--", color="black", alpha=0.5, linewidth=2)

        ax.scatter([threshold], [ks_good_point], color="white", edgecolors="green", s=15)
        ax.scatter([threshold], [ks_bad_point], color="white", edgecolors="red", s=15)
        ax.scatter([threshold], [ks_value], color="white", edgecolors="darkorange", s=15)

        ax.text(threshold, ks_good_point, round(ks_good_point, 4))
        ax.text(threshold, ks_bad_point, round(ks_bad_point, 4))
        ax.text(threshold, ks_value, round(ks_value, 4))

        ax.set_xlabel("thresholds", labelpad=5, loc="center")
        ax.set_title("KS Curve")

        plt.legend(loc=4)
        # plt.show()
        plt.savefig(f"{file_dir}/KS_Curve.png")
        plt.close()

    @classmethod
    def _plot_lift(cls, label, y_prob, file_dir):
        result = pd.DataFrame([label, y_prob]).T
        result.columns = ["target", "proba"]
        result = result.sort_values(["proba", "target"], ascending=False).reset_index()

        del result["index"]
        result.set_index((result.index + 1) / result.shape[0], inplace=True)
        result["bad_sum"] = result["target"].cumsum()
        result["count_sum"] = [i + 1 for i in range(result.shape[0])]
        result["rate"] = result["bad_sum"] / result["count_sum"]
        result["lift"] = result["rate"] / (result["target"].sum() / result.shape[0])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(result["lift"])  # color='darkorange'
        ax.grid(True, linestyle="-.")
        ax.set_title("Lift Curve")
        ax.set_ylabel("lift", labelpad=5, loc="center")
        ax.set_xlabel("predict positive ratio", labelpad=5, loc="center")
        ax.set_xticks([i / 10 for i in range(11)])

        plt.savefig(f"{file_dir}/Lift_Curve.png")
        plt.close()

    @staticmethod
    def plot_f1_score(f1_record, file_dir="./models"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True, linestyle="-.")

        ax.plot(list(range(len(f1_record))), f1_record, ls="-", linewidth=2.0)
        # c='#2d85f0'

        ax.set_title("F1 Record Curve")
        ax.set_ylabel("f1 score", labelpad=5, loc="center")
        ax.set_xlabel("epoch", labelpad=5, loc="center")
        plt.savefig(f"{file_dir}/F1_Curve.png")
        plt.close()

    @staticmethod
    def plot_predict_distribution(y_prob, bins=10, file_dir="./models"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True, linestyle="-.")

        frequency_each, _, _ = ax.hist(x=y_prob, bins=bins, range=(0, 1), color="steelblue")
        indexes = [1 / (2 * bins) + 1 / bins * i for i in range(bins)]
        for x, y in zip(indexes, frequency_each):
            plt.text(x, y + 0.01, f"{y}", horizontalalignment='center')

        ax.set_title("Predict Probability Distribution")
        ax.set_ylabel("Count", labelpad=5, loc="center")
        ax.set_xlabel("Predict Value", labelpad=5, loc="center")
        # plt.show()
        plt.savefig(f"{file_dir}/predict_probability_distribution.png")
        plt.close()

    @staticmethod
    def plot_predict_prob_box(y_prob, file_dir="./models"):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True, linestyle="-.")

        labels = ["predict result"]
        ax.boxplot(y_prob, labels=labels, showmeans=True, meanline=True)

        ax.set_title("Predict Probability Box")
        ax.set_ylabel("value", labelpad=5, loc="center")

        plt.savefig(f"{file_dir}/predict_prob_box.png")
        plt.close()

    @staticmethod
    def plot_residual(residual, file_dir="./models"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True, linestyle="-.")

        ax.plot(list(range(len(residual))), residual, ls="-", linewidth=2.0)
        # c='#2d85f0'

        ax.set_title("Residual Curve")
        ax.set_ylabel("residual value", labelpad=5, loc="center")
        ax.set_xlabel("epoch", labelpad=5, loc="center")
        plt.savefig(f"{file_dir}/residual_analysis.png")
        plt.close()

    @staticmethod
    def plot_iv(iv_dict, file_dir="./models"):
        '''
        _, _, iv_dict = woe.cal_woe()
        woe is in [ActiveWoe(), PassiveWoe()]
        Then iv_dict is the input of this function
        '''
        feature_list = list(iv_dict.keys())
        iv_list = [iv_dict[key] for key in feature_list]
        fig, ax = plt.subplots()
        ax.barh(feature_list, iv_list)
        ax.set_xlabel('IV values')
        ax.set_ylabel('Feature Ids')
        ax.set_title('Feature IV Analysis')
        plt.savefig(f"{file_dir}/iv_analysis.png")
        plt.close()

    @staticmethod
    def plot_bimodal_distribution(data_1, data_2, bins_value=50, file_dir="./models"):

        from scipy.stats import norm

        data = np.concatenate((data_1, data_2))
        counts, bins = np.histogram(data, bins=bins_value)
        bins = bins[:-1] + (bins[1] - bins[0]) / 2
        probs = counts / float(counts.sum())
        pdf_1 = norm.pdf(bins, data_1.mean(), data_1.std())
        pdf_2 = norm.pdf(bins, data_2.mean(), data_2.std())
        plt.plot(bins, probs, label='Data')  # 绘制数据的折线图，添加标签 'Data'
        plt.plot(bins, pdf_1 + pdf_2, label='PDF')  # 绘制两个正态分布曲线之和的折线图，添加标签 'PDF'
        plt.title('Bimodal Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(loc="best")
        plt.savefig(f"{file_dir}/bimodal_distribution.png")
        plt.close()

    @staticmethod
    def plot_ordered_lorenz_curve(label, y_prob, file_dir="./models"):

        label_ranking = np.argsort(label)
        label = label[label_ranking]
        cum_l = np.cumsum(label) / label.sum()
        y_prob = y_prob[label_ranking]

        plt.plot(y_prob, cum_l)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('percentage x')
        plt.ylabel('percentage y')
        plt.title('Lorenz curve')
        plt.legend(loc="best")

        plt.savefig(f"{file_dir}/ordered_lorenz_curve.png")
        plt.close()

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

    @staticmethod
    def plot_trees(tree_strs, file_dir="./models"):
        file_path = os.path.join(file_dir, "trees.txt")

        with open(file_path, "a") as f:
            for tree_id, tree_str in enumerate(tree_strs.values(), 1):
                f.write(f"Tree{tree_id}:\n")
                f.write(tree_str)
                f.write("\n\n")

    @staticmethod
    def plot_importance(
            booster: ActiveTreeParty,
            importance_type: str = "split",
            max_num_features: Optional[int] = 20,
            title: str = "Feature importance",
            xlabel: str = "Importance score",
            ylabel: str = "Features",
            # figsize: Optional[Tuple[float, float]] = None, # raise Cythoning error
            figsize: Optional[tuple] = (14, 8),
            height: float = 0.2,
            xlim: Optional[tuple] = None,
            ylim: Optional[tuple] = None,
            grid: bool = True,
            show_values: bool = True,
            precision: Optional[int] = 3,
            file_dir: str = "./models",
    ) -> Axes:
        """Plot importance based on fitted trees.
        Parameters
        ----------
        booster : ActiveTreeParty or dict
        importance_type : str, default "split"
            How the importance is calculated: either "split", "gain", or "cover"
            * "split" is the number of times a feature appears in trees
            * "gain" is the average gain of splits which use the feature
            * "cover" is the average coverage of splits which use the feature
              where coverage is defined as the number of samples affected by the split
        max_num_features : int, default None
            Maximum number of top features displayed on plot.
            If None, all features will be displayed.
        height : float, default 0.2
            Bar height, passed to ax.barh()
        xlim : tuple, default None
            Tuple passed to axes.xlim()
        ylim : tuple, default None
            Tuple passed to axes.ylim()
        title : str, default "Feature importance"
            Axes title. To disable, pass None.
        xlabel : str, default "F score"
            X axis title label. To disable, pass None.
        ylabel : str, default "Features"
            Y axis title label. To disable, pass None.
        grid : bool, Turn the axes grids on or off.  Default is True (On).
        show_values : bool, default True
            Show values on plot. To disable, pass False.
        precision : int or None, optional (default=3)
            Used to restrict the display of floating point values to a certain precision
        Returns
        -------
        ax : matplotlib Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("You must install matplotlib to plot importance") from e

        # if isinstance(booster, ActiveTreeParty):
        #     importance_info = booster.feature_importances_(importance_type)
        # elif isinstance(booster, dict):
        #     importance_info = booster
        # else:
        #     raise ValueError('tree must be ActivePartyModel or dict instance')
        if isinstance(booster, dict):
            importance_info = booster
        else:
            importance_info = booster.feature_importances_(importance_type)

        # deal feature importance message
        features, values = (
            importance_info["features"],
            importance_info[f"importance_{importance_type}"],
        )
        tuples = sorted(zip(features, values), key=lambda x: x[1])

        if max_num_features is not None and max_num_features > 0:
            tuples = tuples[-max_num_features:]
        features, values = zip(*tuples)

        # set ax
        if figsize is not None:
            Plot._check_not_tuple_of_2_elements(figsize, "figsize")
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ylocs = np.arange(len(values))
        ax.barh(ylocs, values, align="center", height=height)

        gap = min(1, max(values) * 0.02)  # avoid errors when the value is less than 1
        if show_values is True:
            for x, y in zip(values, ylocs):
                ax.text(
                    x + gap,
                    y,
                    Plot._float2str(x, precision) if importance_type == "gain" else x,
                    va="center",
                )

        ax.set_yticks(ylocs)
        ax.set_yticklabels(features)

        # Set the x-axis scope
        if xlim is not None:
            Plot._check_not_tuple_of_2_elements(xlim, "xlim")
        else:
            xlim = (0, max(values) * 1.1)
        ax.set_xlim(xlim)

        # Set the y-axis scope
        if ylim is not None:
            Plot._check_not_tuple_of_2_elements(ylim, "ylim")
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

        plt.savefig(f"{file_dir}/importance.png", pad_inches="tight")
        return ax



if __name__ == "__main__":
    # feature_num = 20
    # features = [f'active_feature{i}' for i in range(feature_num)]
    #
    # importance_type = 'gain'
    # if importance_type == 'split':
    #     values = list(np.random.randint(1, 100, feature_num))
    # else:
    #     values = list(np.random.random(feature_num)+10)
    #     print(values)
    #
    # importance_info = {
    #     'features': list(features),
    #     f'importance_{importance_type}': list(values)
    # }

    # ax = Plot.plot_importance(booster=importance_info,
    #                         importance_type=importance_type,
    #                           figsize=(14, 8))
    # plt.show()
    # labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # probs = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1])
    #
    # # Plot.plot_binary_mertics(labels, probs)
    #
    # train_loss = np.array([0.97, 0.5, 0.25, 0.125, 0.05, 0.04, 0.03])
    # train_auc = np.array([0.5, 0.7, 0.85, 0.9, 0.92, 0.94, 0.95])
    #  test_loss = np.array([0.87, 0.4, 0.25, 0.105, 0.08, 0.07, 0.06])
    # train_loss = np.array([0.97, 0.95, 0.91, 0.9])
    # test_loss = np.array([0.87, 0.85, 0.81, 0.8])

    # Plot.plot_convergence(train_loss, train_auc)
    # Plot.plot_fit(train_loss, test_loss)
    # Plot.plot_train_test_loss(train_loss, test_loss)

    y_label = [1 for _ in range(500)] + [0 for _ in range(500)]
    y_prob = np.random.random(1000)
    print(Evaluate.eval_ks(y_label, y_prob))
    Plot._plot_ks(y_label, y_prob)
    # Plot.plot_predict_distribution(y_prob, bins=10)
    # Plot.plot_regression_metrics(y_prob, y_prob, y_prob, y_prob)
    # Plot.plot_residual(y_label, y_prob)
    # Plot.plot_predict_prob_box(y_prob)



