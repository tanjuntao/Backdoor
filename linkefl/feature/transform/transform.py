import numpy as np
from sklearn import preprocessing
import torch

from .base import BaseTransform
from linkefl.common.const import Const


class Scale(BaseTransform):
    def __init__(self, role):
        super(Scale, self).__init__()
        self.role = role

    def __call__(self, dataset):
        start_col = 2 if self.role == Const.ACTIVE_NAME else 1
        if isinstance(dataset, np.ndarray):
            scaled_feats = preprocessing.scale(dataset[:, start_col:], copy=False)
            dataset[:, start_col:] = scaled_feats
        elif isinstance(dataset, torch.Tensor):
            scaled_feats = preprocessing.scale(dataset[:, start_col:].numpy(), copy=False)
            dataset[:, start_col:] = torch.from_numpy(scaled_feats)
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')

        return dataset


class Normalize(BaseTransform):
    def __init__(self, role, norm=Const.L2):
        super(Normalize, self).__init__()
        self.role = role
        self.norm = norm

    def __call__(self, dataset):
        start_col = 2 if self.role == Const.ACTIVE_NAME else 1
        if isinstance(dataset, np.ndarray):
            normalized_feats = preprocessing.normalize(dataset[:, start_col:],
                                                       norm=self.norm,
                                                       copy=False)
            dataset[:, start_col:] = normalized_feats
        elif isinstance(dataset, torch.Tensor):
            normalized_feats = preprocessing.normalize(dataset[:, start_col:].numpy(),
                                                       norm=self.norm,
                                                       copy=False)
            dataset[:, start_col:] = torch.from_numpy(normalized_feats)
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')

        return dataset


class ParseLabel(BaseTransform):
    def __init__(self, role, neg_label=0):
        super(ParseLabel, self).__init__()
        self.role = role
        self.neg_label = neg_label

    def __call__(self, dataset):
        has_label = True if self.role == Const.ACTIVE_NAME else False
        if isinstance(dataset, np.ndarray):
            if has_label:
                labels = dataset[:, 1]
                labels[labels == -1] = self.neg_label
                dataset[:, 1] = labels
            else:
                pass # no need to parse label for passive party
        elif isinstance(dataset, torch.Tensor):
            if has_label:
                labels = dataset[:, 1]
                labels[labels == -1] = labels
                dataset[:, 1] = labels
            else:
                pass
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')

        return dataset


class AddIntercept(BaseTransform):
    def __init__(self, role):
        super(AddIntercept, self).__init__()
        self.role = role

    def __call__(self, dataset):
        has_label = True if self.role == Const.ACTIVE_NAME else False
        if isinstance(dataset, np.ndarray):
            if has_label:
                n_samples = dataset.shape[0]
                dataset = np.c_[dataset, np.ones(n_samples)]
            else:
                pass # no need to append an intercept column to passive party
        elif isinstance(dataset, torch.Tensor):
            if has_label:
                n_samples = dataset.shape[0]
                dataset = torch.cat((dataset, torch.ones(n_samples)),
                                    dim=1)
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')

        return dataset


class Compose(BaseTransform):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        assert type(transforms) == list, 'Compose can only take a list as parameter'
        for transform in transforms:
            assert isinstance(transform, BaseTransform), \
                'each element in Compose should be an instance of BaseTransform'

        self.transforms = transforms

    def __call__(self, dataset):
        for transform in self.transforms:
            dataset = transform(dataset)
        return dataset
