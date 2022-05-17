import numpy as np
from sklearn import preprocessing
import torch

from linkefl.common.const import Const
from linkefl.dataio import BaseDataset, NumpyDataset, TorchDataset


def scale(dataset):
    if isinstance(dataset, NumpyDataset):
        scaled_feats = preprocessing.scale(dataset.features)
        if dataset.has_label:
            dataset.np_dataset[:, 2:] = scaled_feats
        else:
            dataset.np_dataset[:, 1:] = scaled_feats
    elif isinstance(dataset, TorchDataset):
        scaled_feats = preprocessing.scale(dataset.features.numpy())
        if dataset.has_label:
            dataset.torch_dataset[:, 2:] = torch.from_numpy(scaled_feats)
        else:
            dataset.torch_dataset[:, 1:] = torch.from_numpy(scaled_feats)
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    return dataset


def normalize(dataset, norm=Const.L2):
    if isinstance(dataset, NumpyDataset):
        normalized_feats = preprocessing.normalize(dataset.features, norm=norm)
        if dataset.has_label:
            dataset.np_dataset[:, 2:] = normalized_feats
        else:
            dataset.np_dataset[:, 1:] = normalized_feats
    elif isinstance(dataset, TorchDataset):
        normalized_feats = preprocessing.normalize(dataset.features.numpy(), norm=norm)
        if dataset.has_label:
            dataset.torch_dataset[:, 2:] = torch.from_numpy(normalized_feats)
        else:
            dataset.torch_dataset[:, 1:] = torch.from_numpy(normalized_feats)
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    return dataset


def parse_label(dataset):
    if isinstance(dataset, NumpyDataset):
        if dataset.has_label:
            labels = dataset.labels
            labels[labels == -1] = 0
            dataset.np_dataset[:, 1] = labels
        else:
            pass # no need to parse label for passive pary
    elif isinstance(dataset, TorchDataset):
        if dataset.has_label:
            labels = dataset.labels
            labels[labels == -1] = 0
            dataset.torch_dataset[:, 1] = labels
        else:
            pass
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    return dataset


def add_intercept(dataset):
    if isinstance(dataset, NumpyDataset):
        if dataset.has_label:
            n_samples = dataset.n_samples
            new_np_dataset = np.c_[dataset.np_dataset, np.zeros(n_samples)]
            dataset.set_dataset(new_np_dataset)
        else:
            pass # no need to append an intercept column for passive party
    elif isinstance(dataset, TorchDataset):
        if dataset.has_label:
            n_samples = dataset.n_samples
            new_torch_dataset = torch.cat((dataset.torch_dataset, torch.zeros(n_samples)),
                                          dim=1)
            dataset.set_dataset(new_torch_dataset)
        else:
            pass
    else:
        raise TypeError('dataset should be an instance of'
                        'NumpyDataset or TorchDataset')

    return dataset