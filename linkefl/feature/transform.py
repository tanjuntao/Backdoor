import copy

import numpy as np
from sklearn import preprocessing
import torch

from linkefl.common.const import Const
from linkefl.dataio import BaseDataset, NumpyDataset, TorchDataset


def scale(dataset):
    if isinstance(dataset, NumpyDataset):
        scaled_feats = preprocessing.scale(dataset.features)
        new_dataset = copy.deepcopy(dataset.get_dataset())
        if dataset.has_label:
            new_dataset[:, 2:] = scaled_feats
        else:
            new_dataset[:, 1:] = scaled_feats
    elif isinstance(dataset, TorchDataset):
        scaled_feats = preprocessing.scale(dataset.features.numpy())
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
    if isinstance(dataset, NumpyDataset):
        normalized_feats = preprocessing.normalize(dataset.features, norm=norm)
        new_dataset = copy.deepcopy(dataset.get_dataset())
        if dataset.has_label:
            new_dataset[:, 2:] = normalized_feats
        else:
            new_dataset[:, 1:] = normalized_feats
    elif isinstance(dataset, TorchDataset):
        normalized_feats = preprocessing.normalize(dataset.features.numpy(), norm=norm)
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
    if isinstance(dataset, NumpyDataset):
        if dataset.has_label:
            labels = dataset.labels
            labels[labels == -1] = neg_label
            new_dataset = copy.deepcopy(dataset.get_dataset())
            new_dataset[:, 1] = labels
            dataset.set_dataset(new_dataset)
        else:
            pass # no need to parse label for passive pary
    elif isinstance(dataset, TorchDataset):
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
    if isinstance(dataset, NumpyDataset):
        if dataset.has_label:
            n_samples = dataset.n_samples
            new_np_dataset = np.c_[dataset.get_dataset(), np.ones(n_samples)]
            dataset.set_dataset(new_np_dataset)
        else:
            pass # no need to append an intercept column for passive party
    elif isinstance(dataset, TorchDataset):
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