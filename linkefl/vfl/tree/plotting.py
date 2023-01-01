import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Any
from scipy.stats import scoreatpercentile
from PrettyPrint import PrettyPrintTree
from sklearn.metrics import precision_recall_curve, roc_curve

# TODO：fix ereor "circular import"
# import linkefl

Axes = Any              # real type is matplotlib.axes.Axes
GraphvizSource = Any    # real type is graphviz.Source
ActiveTreeParty = Any   # real type is linkefl.vfl.tree.ActiveTreeParty

class Plot(object):
    def __init__(self):
        pass

    @staticmethod
    def plot_trees(tree_strs, file_dir="./models"):
        file_path = os.path.join(file_dir, "trees.txt")

        with open(file_path, "a") as f:
            for tree_id, tree_str in enumerate(tree_strs.values(), 1):
                f.write(f"Tree{tree_id}:\n")
                f.write(tree_str)

    @staticmethod
    def plot_importance(booster: ActiveTreeParty,
                        importance_type: str = "split",
                        file_dir='./models'):
        ax = plot_importance(booster, importance_type)
        plt.savefig(f"{file_dir}/importance.png")
        plt.close()

    @staticmethod
    def plot_train_test_loss(train_loss, test_loss, file_dir="./models"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(len(train_loss)), train_loss, label='train_loss')  # color='darkorange'
        ax.plot(np.arange(len(test_loss)), test_loss, label='test_loss')
        ax.grid(True, linestyle='-.')
        ax.set_title('train_test_loss')
        ax.set_ylabel('loss', labelpad=5, loc='center')
        ax.set_xlabel('epoch', labelpad=5, loc='center')
        plt.legend(loc='best')

        plt.savefig(f'{file_dir}/train_test_loss.png')
        plt.close()

    @staticmethod
    def plot_train_test_auc(train_auc, test_auc, file_dir="./models"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(len(train_auc)), train_loss, label='train_auc')  # color='darkorange'
        ax.plot(np.arange(len(test_auc)), test_loss, label='test_auc')
        ax.grid(True, linestyle='-.')
        ax.set_title('train_test_auc')
        ax.set_ylabel('loss', labelpad=5, loc='center')
        ax.set_xlabel('epoch', labelpad=5, loc='center')
        plt.legend(loc='best')

        plt.savefig(f'{file_dir}/train_test_auc.png')
        plt.close()

    @classmethod
    def plot_binary_mertics(cls, labels, y_probs, file_dir: str='./models'):
        cls._plot_pr(labels, y_probs, file_dir)
        cls._plot_roc(labels, y_probs, file_dir)
        cls._plot_ks(labels, y_probs, file_dir)
        cls._plot_lift(labels, y_probs, file_dir)

    @classmethod
    def _plot_pr(cls, label, y_prob, file_dir):
        precision, recall, thresholds_pr = precision_recall_curve(label, y_prob)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(recall, precision, ls='-', linewidth=2.0)
        ax.grid(True, linestyle='-.')
        ax.set_xlabel('recall', labelpad=5, loc='center')
        ax.set_ylabel('precision', labelpad=5, loc='center')
        ax.set_title('PR Curve')

        plt.savefig(f'{file_dir}/PR_Curve.png')
        plt.close()

    @classmethod
    def _plot_roc(cls, label, y_prob, file_dir):
        fpr, tpr, thresholds_roc = roc_curve(label, y_prob)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(fpr, tpr, ls='-', linewidth=2.0)
        ax.grid(True, linestyle='-.')
        ax.set_xlabel('false positive rate', labelpad=5, loc='center')
        ax.set_ylabel('true positive rate', labelpad=5, loc='center')
        ax.set_title('ROC Curve')

        plt.savefig(f'{file_dir}/ROC_Curve.png')
        plt.close()

    @classmethod
    def _plot_ks(cls, label, y_prob, file_dir):
        fpr, tpr, thresholds = roc_curve(label, y_prob)
        fpr, tpr, thresholds = fpr[::-1], tpr[::-1], thresholds[::-1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(thresholds, tpr, ls='-', linewidth=2.0)
        ax.plot(thresholds, fpr, ls='-', linewidth=2.0)
        ax.plot(thresholds, tpr - fpr, ls='-', linewidth=2.0)
        ax.grid(True, linestyle='-.')
        ax.set_xlabel('threshold', labelpad=5, loc='center')
        ax.set_title('KS Curve')

        plt.legend(['tpr', 'fpr', 'tpr-fpr'])
        plt.savefig(f'{file_dir}/KS_Curve.png')
        plt.close()

    @classmethod
    def _plot_lift(cls, label, y_prob, file_dir):
        result = pd.DataFrame([label, y_prob]).T
        result.columns = ['target', 'proba']
        result = result.sort_values(['proba', 'target'], ascending=False).reset_index()

        del result['index']
        result.set_index((result.index + 1) / result.shape[0], inplace=True)
        result['bad_sum'] = result['target'].cumsum()
        result['count_sum'] = [i + 1 for i in range(result.shape[0])]
        result['rate'] = result['bad_sum'] / result['count_sum']
        result['lift'] = result['rate'] / (result['target'].sum() / result.shape[0])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(result['lift'])   # color='darkorange'
        ax.grid(True, linestyle='-.')
        ax.set_title('Lift Curve')
        ax.set_ylabel('lift', labelpad=5, loc='center')
        ax.set_xlabel('predict positive ratio', labelpad=5, loc='center')
        ax.set_xticks([i / 10 for i in range(11)])

        plt.savefig(f'{file_dir}/Lift_Curve.png')
        plt.close()


def tree_to_str(tree, tree_structure):
    """

    Args:
        tree: DecisionTree
        tree_structure:

    Returns:
        str for tree structure.
    """
    root = tree.root
    _prepare_print_val(tree, root)

    orientation = PrettyPrintTree.HORIZONTAL if tree_structure == "HORIZONTAL" else PrettyPrintTree.VERTICAL

    pt = PrettyPrintTree(
        get_children=lambda x: x.children if x else [],
        get_val=lambda x: x.print_val if x else "",
        default_orientation = orientation,
        border=True,
        # color=None,
        return_instead_of_print=True,
    )

    tree_str = pt(root)
    return tree_str


def plot_importance(
    booster: ActiveTreeParty,
    importance_type: str = "split",
    max_num_features: Optional[int] = None,
    title: str = "Feature importance",
    xlabel: str = "Importance score",
    ylabel: str = "Features",
    # figsize: Optional[Tuple[float, float]] = None, # raise Cythoning error
    figsize: Optional[tuple] = None,
    height: float = 0.2,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    grid: bool = True,
    show_values: bool = True,
    precision: Optional[int] = 3,
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
        Maximum number of top features displayed on plot. If None, all features will be displayed.
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
        Used to restrict the display of floating point values to a certain precision.
    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('You must install matplotlib to plot importance') from e

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
    features, values = importance_info['features'], importance_info[f'importance_{importance_type}']
    tuples = sorted(zip(features, values), key = lambda x: x[1])

    if max_num_features is not None and max_num_features > 0:
        tuples = tuples[-max_num_features:]
    features, values = zip(*tuples)

    # set ax
    if figsize is not None:
        _check_not_tuple_of_2_elements(figsize, 'figsize')
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height)

    gap = min(1, max(values)*0.02)      # avoid errors when the value is less than 1
    if show_values is True:
        for x, y in zip(values, ylocs):
            ax.text(x + gap, y,
                    _float2str(x, precision) if importance_type == 'gain' else x,
                    va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(features)

    # Set the x-axis scope
    if xlim is not None:
        _check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    # Set the y-axis scope
    if ylim is not None:
        _check_not_tuple_of_2_elements(ylim, 'ylim')
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

    return ax

def _prepare_print_val(tree, root):
    if not root:
        return

    if root.value != None:
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
            print_val += f"feature: encrypt\n"
            print_val += f"threshold: encrypt"

    root.print_val = print_val
    root.children = []
    # print(root.print_val)
    if root.left_branch:
        _prepare_print_val(tree, root.left_branch)
        root.children.append(root.left_branch)
    if root.right_branch:
        _prepare_print_val(tree, root.right_branch)
        root.children.append(root.right_branch)

def _check_not_tuple_of_2_elements(obj: Any, obj_name: str = 'obj') -> None:
    """Check object is not tuple or does not have 2 elements."""
    if not isinstance(obj, tuple) or len(obj) != 2:
        raise TypeError(f"{obj_name} must be a tuple of 2 elements.")

def _float2str(value: float, precision: Optional[int] = None) -> str:
    return (f"{value:.{precision}f}"
            if precision is not None and not isinstance(value, str)
            else str(value))



if __name__ == '__main__':
    # feature_num = 20
    # features = [f'feature{i}' for i in range(feature_num)]
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
    #
    # ax = plot_importance(booster=importance_info,
    #                      importance_type=importance_type)
    # plt.show()
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1])

    # Plot.plot_binary_mertics(labels, probs)

    train_loss = np.array([0.97, 0.5, 0.25, 0.125, 0.05, 0.04, 0.03])
    train_auc = np.array([0.5, 0.7, 0.85, 0.9, 0.92, 0.94, 0.95])
    test_loss = np.array([0.87, 0.4, 0.25, 0.105, 0.08, 0.07, 0.06])

    # Plot.plot_convergence(train_loss, train_auc)
    # Plot.plot_fit(train_loss, test_loss)