import copy
from typing import List, Optional, Union

import pandas as pd
import torch

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset, TorchDataset


def scale(
    dataset: Union[NumpyDataset, TorchDataset]
) -> Union[NumpyDataset, TorchDataset]:
    from linkefl.feature.transform.transform import Scale

    dataset = Scale().fit(dataset, role=dataset.role)
    return dataset


def normalize(
    dataset: Union[NumpyDataset, TorchDataset],
    norm: str = Const.L2,
) -> Union[NumpyDataset, TorchDataset]:
    from linkefl.feature.transform.transform import Normalize

    dataset = Normalize(norm=norm).fit(dataset, role=dataset.role)
    return dataset


def parse_label(
    dataset: Union[NumpyDataset, TorchDataset],
    neg_label: int = 0,
) -> Union[NumpyDataset, TorchDataset]:
    from linkefl.feature.transform.transform import ParseLabel

    dataset = ParseLabel(neg_label=neg_label).fit(dataset, role=dataset.role)
    return dataset


def add_intercept(
    dataset: Union[NumpyDataset, TorchDataset],
) -> Union[NumpyDataset, TorchDataset]:
    from linkefl.feature.transform.transform import AddIntercept

    dataset = AddIntercept().fit(dataset, role=dataset.role)
    return dataset


def one_hot(
    dataset: Union[NumpyDataset, TorchDataset],
    idxes: Optional[List[int]] = None,
) -> Union[NumpyDataset, TorchDataset]:
    from linkefl.feature.transform.transform import OneHot

    dataset = OneHot(idxes=idxes).fit(dataset, role=dataset.role)
    return dataset


def bin(dataset, bin_features, bin_methods, para):
    """Feature binning on the given feature column.

    Parameters
    ----------
    dataset : NumpyDataset / TorchDataset
        Original complete data set.

    bin_features: list
        The position of the column to be binned.

    bin_methods: string or list
        The method corresponds to the column to be binned.
        (ed for equal distance / ef for equal frequency)
        When its type is a string,
        it means that all specified columns will be binned
        using the method represented by this string.
        When its type is a list,
        it means that each specified column will be binned
        using the corresponding method within the list.

    para: list
        The number of the bins for the column to be binned.

    Returns
    -------
    dataset: NumpyDataset / TorchDataset

    """
    if isinstance(dataset, NumpyDataset):
        new_dataset = copy.deepcopy(dataset.get_dataset())
        split = dict()
        if not isinstance(bin_methods, list):
            bin_methods = [bin_methods] * len(bin_features)

        for i in range(len(bin_features)):
            if bin_methods[i] == "equalizer":
                new_dataset[:, bin_features[i]], split[bin_features[i]] = pd.cut(
                    new_dataset[:, bin_features[i]], para[i], labels=False, retbins=True
                )
                split[bin_features[i]] = split[bin_features[i]][1:-1]
            elif bin_methods[i] == "quantiler":
                new_dataset[:, bin_features[i]], split[bin_features[i]] = pd.qcut(
                    new_dataset[:, bin_features[i]], para[i], labels=False, retbins=True
                )
                split[bin_features[i]] = split[bin_features[i]][1:-1]
            else:
                raise TypeError("Method %d should be equalizer or quantiler" % i)

    elif isinstance(dataset, TorchDataset):
        if not isinstance(bin_methods, list):
            bin_methods = [bin_methods] * len(bin_features)
        new_dataset = copy.deepcopy(dataset.get_dataset())
        split = dict()
        for i in range(len(bin_features)):
            if bin_methods[i] == "equalizer":
                temp_dataset, split[bin_features[i]] = pd.cut(
                    new_dataset[:, bin_features[i]].numpy(),
                    para[i],
                    labels=False,
                    retbins=True,
                )
                new_dataset[:, bin_features[i]] = torch.from_numpy(temp_dataset)
                split[bin_features[i]] = split[bin_features[i]][1:-1]
            elif bin_methods[i] == "quantiler":
                temp_dataset, split[bin_features[i]] = pd.qcut(
                    new_dataset[:, bin_features[i]].numpy(),
                    para[i],
                    labels=False,
                    retbins=True,
                )
                new_dataset[:, bin_features[i]] = torch.from_numpy(temp_dataset)
                split[bin_features[i]] = split[bin_features[i]][1:-1]
            else:
                raise TypeError("Method %d should be equalizer or quantiler" % i)
    else:
        raise TypeError("dataset should be an instance ofNumpyDataset or TorchDataset")

    dataset.set_dataset(new_dataset)  # update status of dataset object
    setattr(dataset, "bin_split", split)  # set the bin_split attr of dataset object

    return dataset
