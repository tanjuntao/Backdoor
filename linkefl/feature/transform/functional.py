import copy

import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch

import linkefl
from linkefl.common.const import Const

# the following import syntax will cause circular import error
# from linkefl.dataio import BaseDataset, NumpyDataset, TorchDataset


def scale(dataset):
    # 1. maximum memory usage is three copies of the original dataset.
    # 2. use abosulute import to avoid circular import error 
    if isinstance(dataset, linkefl.dataio.NumpyDataset):
        # assign copy=False to save memory usage
        scaled_feats = preprocessing.scale(dataset.features, copy=False)
        new_dataset = copy.deepcopy(dataset.get_dataset())
        if dataset.has_label:
            new_dataset[:, 2:] = scaled_feats
        else:
            new_dataset[:, 1:] = scaled_feats
    elif isinstance(dataset, linkefl.dataio.TorchDataset):
        scaled_feats = preprocessing.scale(dataset.features.numpy().astype(np.float64), copy=False)
        new_dataset = copy.deepcopy(dataset.get_dataset())
        if dataset.has_label:
            new_dataset[:, 2:] = torch.from_numpy(scaled_feats)
        else:
            new_dataset[:, 1:] = torch.from_numpy(scaled_feats)
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    del scaled_feats # save memory
    dataset.set_dataset(new_dataset) # update status of dataset object

    return dataset


def normalize(dataset, norm=Const.L2):
    # maximum memory usage is three copies of the original dataset
    if isinstance(dataset, linkefl.dataio.NumpyDataset):
        normalized_feats = preprocessing.normalize(dataset.features,
                                                   norm=norm,
                                                   copy=False)
        new_dataset = copy.deepcopy(dataset.get_dataset())
        if dataset.has_label:
            new_dataset[:, 2:] = normalized_feats
        else:
            new_dataset[:, 1:] = normalized_feats
    elif isinstance(dataset, linkefl.dataio.TorchDataset):
        normalized_feats = preprocessing.normalize(dataset.features.numpy(),
                                                   norm=norm,
                                                   copy=False)
        new_dataset = copy.deepcopy(dataset.get_dataset())
        if dataset.has_label:
            new_dataset[:, 2:] = torch.from_numpy(normalized_feats)
        else:
            new_dataset[:, 1:] = torch.from_numpy(normalized_feats)
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    del normalized_feats
    dataset.set_dataset(new_dataset)

    return dataset


def parse_label(dataset, neg_label=0):
    # maximum memory usage is two copies of the original dataset
    if isinstance(dataset, linkefl.dataio.NumpyDataset):
        if dataset.has_label:
            labels = dataset.labels
            labels[labels == -1] = neg_label
            new_dataset = copy.deepcopy(dataset.get_dataset())
            new_dataset[:, 1] = labels
            dataset.set_dataset(new_dataset)
        else:
            pass # no need to parse label for passive pary
    elif isinstance(dataset, linkefl.dataio.TorchDataset):
        if dataset.has_label:
            labels = dataset.labels
            labels[labels == -1] = neg_label
            new_dataset = copy.deepcopy(dataset.get_dataset())
            new_dataset[:, 1] = labels
            dataset.set_dataset(new_dataset)
        else:
            pass
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    return dataset


def add_intercept(dataset):
    # maximum memory usage is two copies of the original dataset
    if isinstance(dataset, linkefl.dataio.NumpyDataset):
        if dataset.has_label:
            n_samples = dataset.n_samples
            new_np_dataset = np.c_[dataset.get_dataset(), np.ones(n_samples)]
            dataset.set_dataset(new_np_dataset)
        else:
            pass # no need to append an intercept column for passive party
    elif isinstance(dataset, linkefl.dataio.TorchDataset):
        if dataset.has_label:
            n_samples = dataset.n_samples
            new_torch_dataset = torch.cat((dataset.get_dataset(), torch.ones(n_samples)),
                                          dim=1)
            dataset.set_dataset(new_torch_dataset)
        else:
            pass
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    return dataset


def one_hot(dataset, index=[]):
    """One-hot coding of discrete data in the data.

    Parameters
    ----------
    dataset : NumpyDataset
        Original complete data set.
        dataset._np_dataset: DataFrame

    index: list
        The position of the column to be one-hot encoded, not the label of the column.

    Returns
    -------
    dataset: NumpyDataset
        The complete data set after one-hot coding, the number of columns may be increased.
        dataset._np_dataset: Numpy.ndarray
    
    """
    if isinstance(dataset, linkefl.dataio.NumpyDataset):
        n_features = dataset.n_features
        for i in range(n_features):
            if i in index:
                # Require to do One-hot coding
                data = pd.DataFrame(dataset.get_dataset().iloc[:, i:i+1])
                new_np_data = np.array(pd.get_dummies(data))
                if i == 0:
                    new_np_dataset = new_np_data
                else:
                    new_np_dataset = np.c_[new_np_dataset, new_np_data]
            else:
                # No require to do One-hot coding
                new_np_data  = np.array(pd.DataFrame(dataset.get_dataset().iloc[:, i:i+1]))
                if i == 0:
                    new_np_dataset = new_np_data
                else:
                    new_np_dataset = np.c_[new_np_dataset, new_np_data]
        dataset.set_dataset(new_np_dataset)
    elif isinstance(dataset, linkefl.dataio.TorchDataset):
        n_features = dataset.n_features
        for i in range(n_features):
            if i in index:
                # Require to do One-hot coding
                data = pd.DataFrame(dataset.get_dataset().iloc[:, i:i+1])
                new_torch_data = torch.from_numpy(np.array(pd.get_dummies(data))) 
                if i == 0:
                    new_torch_dataset = new_torch_data
                else:
                    new_torch_dataset = torch.cat((new_torch_data, new_torch_data), dim=1)
            else:
                # No require to do One-hot coding
                new_torch_data  = torch.from_numpy(np.array(pd.DataFrame(dataset.get_dataset().iloc[:, i:i+1])))
                if i == 0:
                    new_torch_dataset = new_torch_data
                else:
                    new_torch_dataset = torch.cat((new_torch_data, new_torch_data), dim=1)
        dataset.set_dataset(new_torch_dataset)
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')
    
    
    return dataset


def bin(dataset, bin_features, bin_methods, para):
    """Feature binning on the given feature column.

        Parameters
        ----------
        dataset : NumpyDataset / TorchDataset
            Original complete data set.

        bin_features: list
            The position of the column to be binned.

        bin_methods: list
            The method corresponds to the column to be binned. (ed for equal distance / ef for equal frequency)

        para: list
            The number of the bins for the column to be binned.

        Returns
        -------
        dataset: NumpyDataset / TorchDataset

        """
    if isinstance(dataset, linkefl.dataio.NumpyDataset):
        new_dataset = copy.deepcopy(dataset.get_dataset())
        for i in range(len(bin_features)):
            if bin_methods[i] == 'ed':
                new_dataset[:, bin_features[i]] = pd.cut(new_dataset[:, bin_features[i]], para[i], labels=False)
            elif bin_methods[i] == 'ef':
                new_dataset[:, bin_features[i]] = pd.qcut(new_dataset[:, bin_features[i]], para[i], labels=False)
            else:
                raise TypeError('Method %d should be ed or ef' % i)
    elif isinstance(dataset, linkefl.dataio.TorchDataset):
        new_dataset = copy.deepcopy(dataset.get_dataset())
        for i in range(len(bin_features)):
            if bin_methods[i] == 'ed':
                new_dataset[:, bin_features[i]] = torch.from_numpy(pd.cut(new_dataset[:, bin_features[i]].numpy(), para[i], labels=False))
            elif bin_methods[i] == 'ef':
                new_dataset[:, bin_features[i]] = torch.from_numpy(pd.qcut(new_dataset[:, bin_features[i]].numpy(), para[i], labels=False))
            else:
                raise TypeError('Method %d should be ed or ef' % i)
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    dataset.set_dataset(new_dataset) # update status of dataset object

    return dataset


