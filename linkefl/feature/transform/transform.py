from typing import List, Any

import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch

from .base import BaseTransform
from linkefl.common.const import Const
from linkefl import dataio


def wrapper_for_dataset(func):
    def wrapper(self, dataset, role):
        if isinstance(dataset, dataio.NumpyDataset) or isinstance(dataset, dataio.TorchDataset):
            raw_dataset = dataset.get_dataset()
            raw_dataset = func(self, raw_dataset, role)
            dataset.set_dataset(raw_dataset)
        else:
            dataset = func(self, dataset, role)
        return dataset
    return wrapper


class Scale(BaseTransform):
    def __init__(self):
        super(Scale, self).__init__()

    @wrapper_for_dataset
    def __call__(self, dataset, role):
        start_col = 2 if role == Const.ACTIVE_NAME else 1
        if isinstance(dataset, np.ndarray):
            # assign copy=False to save memory usage
            scaled_feats = preprocessing.scale(dataset[:, start_col:], copy=False)
            dataset[:, start_col:] = scaled_feats
        elif isinstance(dataset, torch.Tensor):
            scaled_feats = preprocessing.scale(dataset[:, start_col:].numpy(), copy=False)
            dataset[:, start_col:] = torch.from_numpy(scaled_feats)
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')

        return dataset


class Normalize(BaseTransform):
    def __init__(self, norm=Const.L2):
        super(Normalize, self).__init__()
        self.norm = norm

    @wrapper_for_dataset
    def __call__(self, dataset, role):
        start_col = 2 if role == Const.ACTIVE_NAME else 1
        if isinstance(dataset, np.ndarray):
            # assign copy=False to save memory usage
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
    def __init__(self, neg_label=0):
        super(ParseLabel, self).__init__()
        self.neg_label = neg_label

    @wrapper_for_dataset
    def __call__(self, dataset, role):
        has_label = True if role == Const.ACTIVE_NAME else False
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
                labels[labels == -1] = self.neg_label
                dataset[:, 1] = labels
            else:
                pass
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')

        return dataset


class AddIntercept(BaseTransform):
    def __init__(self):
        super(AddIntercept, self).__init__()

    @wrapper_for_dataset
    def __call__(self, dataset, role):
        has_label = True if role == Const.ACTIVE_NAME else False
        if isinstance(dataset, np.ndarray):
            if has_label:
                n_samples = dataset.shape[0]
                dataset = np.c_[dataset, np.ones(n_samples)]
            else:
                pass # no need to append an intercept column for passive party dataset
        elif isinstance(dataset, torch.Tensor):
            if has_label:
                n_samples = dataset.shape[0]
                dataset = torch.cat((dataset, torch.ones(n_samples)), dim=1)
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
    def __init__(self, idxes):
        super(OneHot, self).__init__()
        if type(idxes) != list or not idxes:
            raise ValueError('idxes should be an non-empty Python list')
        self.idxes = idxes

    # TODO: should be the same as other transformers
    def __call__(self, df, role):
        offset = 2 if role == Const.ACTIVE_NAME else 1
        n_features = df.shape[1] - offset

        all_idxes = list(range(n_features))
        remain_idxes = list(set(all_idxes) - set(self.idxes))
        sub_dfs = list() # a list composed of sub dataframes
        sub_dfs.append(df[:, list(range(offset))]) # store ids or ids+labels
        # store non one-hot features
        sub_dfs.append(df[:, [idx+offset for idx in remain_idxes]])

        for idx in self.idxes:
            if df[idx+offset].dtype in ('float32', 'float64'):
                raise ValueError(f"{idx+offset}-th column has a dtype of float,"
                                 f"which should not be applied by OneHot.")
            dummy_df = pd.get_dummies(df[idx+offset], prefix=str(idx+offset))
            sub_dfs.append(dummy_df)

        # set copy=False to save memory
        final_df = pd.concat(sub_dfs, axis=1, copy=False)
        return final_df


        # if datatype == 'numpy.ndarray':
        #
        #
        #     n_features = dataset.shape[1]
        #     for i in range(n_features):
        #         if i in self.index:
        #             # Require to do One-hot coding
        #             data = pd.DataFrame(dataset.iloc[:, i:i+1])
        #             new_data = np.array(pd.get_dummies(data))
        #             if i == 0:
        #                 new_dataset = new_data
        #             else:
        #                 new_dataset = np.c_[new_dataset, new_data]
        #         else:
        #             # No require to do One-hot coding
        #             new_data  = np.array(pd.DataFrame(dataset.iloc[:, i:i+1]))
        #             if i == 0:
        #                 new_dataset = new_data
        #             else:
        #                 new_dataset = np.c_[new_dataset, new_data]
        #
        # elif datatype == 'torch.Tensor':
        #     n_features = dataset.shape[1]
        #     for i in range(n_features):
        #         if i in self.index:
        #             # Require to do One-hot coding
        #             data = pd.DataFrame(dataset.iloc[:, i:i+1])
        #             new_data = torch.from_numpy(np.array(pd.get_dummies(data)))
        #             if i == 0:
        #                 new_dataset = new_data
        #             else:
        #                 new_dataset = torch.cat((new_dataset, new_data), dim=1)
        #         else:
        #             # No require to do One-hot coding
        #             new_data  = torch.from_numpy(np.array(pd.DataFrame(dataset.iloc[:, i:i+1])))
        #             if i == 0:
        #                 new_dataset = new_data
        #             else:
        #                 new_dataset = torch.cat((new_dataset, new_data), dim=1)
        #
        # else:
        #     raise TypeError('The target type of the dataset should be numpy.ndarray or torch.Tensor')
        #
        # return new_dataset


class Bin(BaseTransform):
    def __init__(self, bin_features=None, bin_methods=None, para=None):
        super(Bin, self).__init__()
        self.bin_features = bin_features if bin_features is not None else []
        self.bin_methods = bin_methods if bin_methods is not None else []
        if isinstance(bin_methods, list) == False:
            self.bin_methods = [bin_methods] * len(bin_features)
        self.para = para if para is not None else []
        self.split = dict()

    # TODO: should be the same as other transformers
    # TODO: 仅返回 dataset 一个参数
    def __call__(self, dataset):
        if isinstance(dataset, np.ndarray):
            for i in range(len(self.bin_features)):
                if self.bin_methods[i] == 'equalizer':
                    dataset[:, self.bin_features[i]], self.split[self.bin_features[i]] = \
                        pd.cut(dataset[:, self.bin_features[i]], self.para[i], labels=False, retbins=True)
                    self.split[self.bin_features[i]] = self.split[self.bin_features[i]][1: -1]
                elif self.bin_methods[i] == 'quantiler':
                    dataset[:, self.bin_features[i]], self.split[self.bin_features[i]] = \
                        pd.qcut(dataset[:, self.bin_features[i]], self.para[i], labels=False, retbins=True)
                    self.split[self.bin_features[i]] = self.split[self.bin_features[i]][1: -1]
                else:
                    raise TypeError('Method %d should be ed or ef' % i)
        elif isinstance(dataset, torch.Tensor):
            for i in range(len(self.bin_features)):
                if self.bin_methods[i] == 'equalizer':
                    temp_dataset, self.split[self.bin_features[i]] = \
                        pd.cut(dataset[:, self.bin_features[i]].numpy(), self.para[i], labels=False, retbins=True)
                    dataset[:, self.bin_features[i]] = torch.from_numpy(temp_dataset)
                    self.split[self.bin_features[i]] = self.split[self.bin_features[i]][1: -1]
                elif self.bin_methods[i] == 'quantiler':
                    temp_dataset, self.split[self.bin_features[i]] = \
                        pd.qcut(dataset[:, self.bin_features[i]].numpy(), self.para[i], labels=False, retbins=True)
                    dataset[:, self.bin_features[i]] = torch.from_numpy(temp_dataset)
                    self.split[self.bin_features[i]] = self.split[self.bin_features[i]][1: -1]
                else:
                    raise TypeError('Method %d should be ed or ef' % i)
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')

        return dataset, self.split


class Compose(BaseTransform):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        assert type(transforms) == list, 'the type of Compose object should be a Python list'
        for transform in transforms:
            assert isinstance(transform, BaseTransform), \
                'each element in Compose should be an instance of BaseTransform'

        self.transforms = transforms

    @wrapper_for_dataset
    def __call__(self, dataset, role):
        # argument role is just for interface consistency
        flag = True
        for transform in self.transforms:
            if isinstance(transform, Bin):
                dataset, split = transform(dataset)
                flag = False
            else:
                dataset = transform(dataset, role)
        if flag is True:
            return dataset
        else:
            return dataset, split
