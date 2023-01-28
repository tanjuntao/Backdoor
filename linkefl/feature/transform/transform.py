import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

from linkefl.base import BaseTransformComponent
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset, TorchDataset


def wrapper_for_dataset(func):
    def wrapper(self, dataset, role):
        if isinstance(dataset, (NumpyDataset, TorchDataset)):
            raw_dataset = dataset.get_dataset()
            raw_dataset = func(self, raw_dataset, role)
            dataset.set_dataset(raw_dataset)
        else:
            dataset = func(self, dataset, role)
        return dataset

    return wrapper


class Scale(BaseTransformComponent):
    @wrapper_for_dataset
    def __call__(self, dataset, role):
        start_col = 2 if role == Const.ACTIVE_NAME else 1
        if isinstance(dataset, np.ndarray):
            # assign copy=False to save memory usage
            scaled_feats = preprocessing.scale(dataset[:, start_col:], copy=False)
            dataset[:, start_col:] = scaled_feats
        elif isinstance(dataset, torch.Tensor):
            scaled_feats = preprocessing.scale(
                dataset[:, start_col:].numpy(), copy=False
            )
            dataset[:, start_col:] = torch.from_numpy(scaled_feats)
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )

        return dataset


class Normalize(BaseTransformComponent):
    def __init__(self, norm=Const.L2):
        self.norm = norm

    @wrapper_for_dataset
    def __call__(self, dataset, role):
        start_col = 2 if role == Const.ACTIVE_NAME else 1
        if isinstance(dataset, np.ndarray):
            # assign copy=False to save memory usage
            normalized_feats = preprocessing.normalize(
                dataset[:, start_col:], norm=self.norm, copy=False
            )
            dataset[:, start_col:] = normalized_feats
        elif isinstance(dataset, torch.Tensor):
            normalized_feats = preprocessing.normalize(
                dataset[:, start_col:].numpy(), norm=self.norm, copy=False
            )
            dataset[:, start_col:] = torch.from_numpy(normalized_feats)
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )

        return dataset


class ParseLabel(BaseTransformComponent):
    def __init__(self, neg_label=0):
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
                pass  # no need to parse label for passive party
        elif isinstance(dataset, torch.Tensor):
            if has_label:
                labels = dataset[:, 1]
                labels[labels == -1] = self.neg_label
                dataset[:, 1] = labels
            else:
                pass
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )

        return dataset


class AddIntercept(BaseTransformComponent):
    @wrapper_for_dataset
    def __call__(self, dataset, role):
        has_label = True if role == Const.ACTIVE_NAME else False
        if isinstance(dataset, np.ndarray):
            if has_label:
                n_samples = dataset.shape[0]
                dataset = np.c_[dataset, np.ones(n_samples)]
            else:
                pass  # no need to append an intercept column for passive party dataset
        elif isinstance(dataset, torch.Tensor):
            if has_label:
                n_samples = dataset.shape[0]
                dataset = torch.cat((dataset, torch.ones(n_samples)), dim=1)
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )

        return dataset


class OneHot(BaseTransformComponent):
    def __init__(self, idxes=None):
        if idxes is None:
            idxes = []
        self.idxes = idxes

    @wrapper_for_dataset
    def __call__(self, dataset, role):
        df_dataset = pd.DataFrame(dataset)

        offset = 2 if role == Const.ACTIVE_NAME else 1
        n_features = df_dataset.shape[1] - offset

        all_idxes = list(range(n_features))
        remain_idxes = list(set(all_idxes) - set(self.idxes))
        sub_dfs = list()  # a list composed of sub dataframes
        sub_dfs.append(df_dataset.iloc[:, :offset])  # store ids or ids+labels
        # store non one-hot features
        sub_dfs.append(df_dataset.iloc[:, [idx + offset for idx in remain_idxes]])

        for idx in self.idxes:
            # if df_dataset[idx + offset].dtype in ('float32', 'float64'):
            #     raise ValueError(f"{idx+offset}-th column has a dtype of float,"
            #                      f"which should not be applied by OneHot.")
            dummy_df = pd.get_dummies(
                df_dataset[idx + offset], prefix=str(idx + offset)
            )
            sub_dfs.append(dummy_df)

        # set copy=False to save memory
        final_df = pd.concat(sub_dfs, axis=1, copy=False)
        new_dataset = final_df.to_numpy()
        return new_dataset


class Bin(BaseTransformComponent):
    def __init__(self, bin_features=None, bin_methods=None, para=None):
        self.bin_features = bin_features if bin_features is not None else []
        self.bin_methods = bin_methods if bin_methods is not None else []
        if not isinstance(bin_methods, list):
            self.bin_methods = [bin_methods] * len(bin_features)
        self.para = para if para is not None else []
        self.split = dict()

    # TODO: should be the same as other transformers
    # TODO: 仅返回 dataset 一个参数
    def __call__(self, dataset):
        if isinstance(dataset, np.ndarray):
            for i in range(len(self.bin_features)):
                if self.bin_methods[i] == "equalizer":
                    (
                        dataset[:, self.bin_features[i]],
                        self.split[self.bin_features[i]],
                    ) = pd.cut(
                        dataset[:, self.bin_features[i]],
                        self.para[i],
                        labels=False,
                        retbins=True,
                    )
                    self.split[self.bin_features[i]] = self.split[self.bin_features[i]][
                        1:-1
                    ]
                elif self.bin_methods[i] == "quantiler":
                    (
                        dataset[:, self.bin_features[i]],
                        self.split[self.bin_features[i]],
                    ) = pd.qcut(
                        dataset[:, self.bin_features[i]],
                        self.para[i],
                        labels=False,
                        retbins=True,
                    )
                    self.split[self.bin_features[i]] = self.split[self.bin_features[i]][
                        1:-1
                    ]
                else:
                    raise TypeError("Method %d should be ed or ef" % i)
        elif isinstance(dataset, torch.Tensor):
            for i in range(len(self.bin_features)):
                if self.bin_methods[i] == "equalizer":
                    temp_dataset, self.split[self.bin_features[i]] = pd.cut(
                        dataset[:, self.bin_features[i]].numpy(),
                        self.para[i],
                        labels=False,
                        retbins=True,
                    )
                    dataset[:, self.bin_features[i]] = torch.from_numpy(temp_dataset)
                    self.split[self.bin_features[i]] = self.split[self.bin_features[i]][
                        1:-1
                    ]
                elif self.bin_methods[i] == "quantiler":
                    temp_dataset, self.split[self.bin_features[i]] = pd.qcut(
                        dataset[:, self.bin_features[i]].numpy(),
                        self.para[i],
                        labels=False,
                        retbins=True,
                    )
                    dataset[:, self.bin_features[i]] = torch.from_numpy(temp_dataset)
                    self.split[self.bin_features[i]] = self.split[self.bin_features[i]][
                        1:-1
                    ]
                else:
                    raise TypeError("Method %d should be ed or ef" % i)
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )

        return dataset, self.split


class Compose(BaseTransformComponent):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        assert (
            type(transforms) == list
        ), "the type of Compose object should be a Python list"
        for transform in transforms:
            assert isinstance(
                transform, BaseTransformComponent
            ), "each element in Compose should be an instance of BaseTransformComponent"

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
