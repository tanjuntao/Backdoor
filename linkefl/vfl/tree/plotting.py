import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Any, Tuple
# TODO：fix ereor "circular import"
# import linkefl

Axes = Any              # real type is matplotlib.axes.Axes
GraphvizSource = Any    # real type is graphviz.Source
ActiveTreeParty = Any   # real type is linkefl.vfl.tree.ActiveTreeParty

def _check_not_tuple_of_2_elements(obj: Any, obj_name: str = 'obj') -> None:
    """Check object is not tuple or does not have 2 elements."""
    if not isinstance(obj, tuple) or len(obj) != 2:
        raise TypeError(f"{obj_name} must be a tuple of 2 elements.")

def _float2str(value: float, precision: Optional[int] = None) -> str:
    return (f"{value:.{precision}f}"
            if precision is not None and not isinstance(value, str)
            else str(value))

def plot_importance(
    booster: ActiveTreeParty,
    importance_type: str = "split",
    max_num_features: Optional[int] = None,
    title: str = "Feature importance",
    xlabel: str = "Importance score",
    ylabel: str = "Features",
    figsize: Optional[Tuple[float, float]] = None,
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

if __name__ == '__main__':
    feature_num = 20
    features = [f'feature{i}' for i in range(feature_num)]

    importance_type = 'gain'
    if importance_type == 'split':
        values = list(np.random.randint(1, 100, feature_num))
    else:
        values = list(np.random.random(feature_num)+10)
        print(values)

    importance_info = {
        'features': list(features),
        f'importance_{importance_type}': list(values)
    }

    ax = plot_importance(booster=importance_info,
                         importance_type=importance_type)
    plt.show()
