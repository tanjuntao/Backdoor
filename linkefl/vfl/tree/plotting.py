from typing import Optional, Any
from linkefl.vfl.tree import ActiveTreeParty

Axes = Any              # real type is matplotlib.axes.Axes
GraphvizSource = Any    # real type is graphviz.Source

def plot_importance(
    booster: ActiveTreeParty,
    ax: Optional[Axes] = None,
    height: float = 0.2,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    title: str = "Feature importance",
    xlabel: str = "F score",
    ylabel: str = "Features",
    fmap: PathLike = "",
    importance_type: str = "weight",
    max_num_features: Optional[int] = None,
    grid: bool = True,
    show_values: bool = True,
    **kwargs: Any
) -> Axes:
    """Plot importance based on fitted trees.
    Parameters
    ----------
    booster : Booster, XGBModel or dict
        Booster or XGBModel instance, or dict taken by Booster.get_fscore()
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.
    grid : bool, Turn the axes grids on or off.  Default is True (On).
    importance_type : str, default "split"
        How the importance is calculated: either "split", "gain", or "cover"
        * "split" is the number of times a feature appears in a tree
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
    fmap: str or os.PathLike (optional)
        The name of feature map file.
    show_values : bool, default True
        Show values on plot. To disable, pass False.
    kwargs :
        Other keywords passed to ax.barh()
    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('You must install matplotlib to plot importance') from e

    if isinstance(booster, ActiveTreeParty):
        importance = ActiveTreeParty.feature_importances_(importance_type)
    elif isinstance(booster, dict):
        importance = booster
    else:
        raise ValueError('tree must be ActivePartyModel or dict instance')

    if not importance:
        raise ValueError(
            'Booster.get_score() results in empty.  ' +
            'This maybe caused by having all trees as decision dumps.')

    tuples = [(k, importance[k]) for k in importance]
    if max_num_features is not None:
        # pylint: disable=invalid-unary-operand-type
        tuples = sorted(tuples, key=lambda _x: _x[1])[-max_num_features:]
    else:
        tuples = sorted(tuples, key=lambda _x: _x[1])
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    if show_values is True:
        for x, y in zip(values, ylocs):
            ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim) != 2:
            raise ValueError('xlim must be a tuple of 2 elements')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if not isinstance(ylim, tuple) or len(ylim) != 2:
            raise ValueError('ylim must be a tuple of 2 elements')
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