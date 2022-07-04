import numpy as np
import pandas as pd
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


class OneHot(BaseTransform):
    """
    An example of how to use this class:
        data = pd.DataFrame(data=[[1, 'fe', 'y'],[2, 'ma', 'n'],[3, 'ma', 'n']] )
        data.to_csv(r"D:/project/LinkeFL/linkefl/data/test1.csv", sep=',', header=False, index=False)

        abs_path = r"D:/project/LinkeFL/linkefl/data/test1.csv"
        transform = Compose([OneHot(target_datatype='numpy.ndarray', index=[1, 2])])
        
        dataset = NumpyDataset(role=Const.ACTIVE_NAME,
                    abs_path=abs_path,
                    transform=transform)

    existing_dataset:
        0   1   2
    0   1   fe  y
    1   2   ma  n
    2   3   ma  n
    dataset:
    tensor([[1, 1, 0, 0, 1],
             2, 0, 1, 1, 0],
             3, 0, 1, 1, 0]])
    """
    def __init__(self, target_datatype, index=[]):
        super(OneHot, self).__init__()
        self.index = index
        self.target_datatype = target_datatype

    def __call__(self, dataset):
        if self.target_datatype == 'numpy.ndarray':
            n_features = dataset.shape[1]
            for i in range(n_features):
                if i in self.index:
                    # Require to do One-hot coding
                    data = pd.DataFrame(dataset.iloc[:, i:i+1])
                    new_data = np.array(pd.get_dummies(data))
                    if i == 0:
                        new_dataset = new_data
                    else:
                        new_dataset = np.c_[new_dataset, new_data]
                else:
                    # No require to do One-hot coding
                    new_data  = np.array(pd.DataFrame(dataset.iloc[:, i:i+1]))
                    if i == 0:
                        new_dataset = new_data
                    else:
                        new_dataset = np.c_[new_dataset, new_data]
        elif self.target_datatype == 'torch.Tensor':
            n_features = dataset.shape[1]
            for i in range(n_features):
                if i in self.index:
                    # Require to do One-hot coding
                    data = pd.DataFrame(dataset.iloc[:, i:i+1])
                    new_data = torch.from_numpy(np.array(pd.get_dummies(data)))
                    if i == 0:
                        new_dataset = new_data
                    else:
                        new_dataset = torch.cat((new_dataset, new_data), dim=1)
                else:
                    # No require to do One-hot coding
                    new_data  = torch.from_numpy(np.array(pd.DataFrame(dataset.iloc[:, i:i+1])))
                    if i == 0:
                        new_dataset = new_data
                    else:
                        new_dataset = torch.cat((new_dataset, new_data), dim=1)
        else:
            raise TypeError('The target type of the dataset should be numpy.ndarray or torch.Tensor')
        
        return new_dataset


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