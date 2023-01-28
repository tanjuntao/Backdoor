from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch


class Basewoe(ABC):
    def __init__(self, dataset, idxes):
        """Woe encodes the specified features and calculates their iv values.

        Parameters
        ----------
        dataset : NumpyDataset / TorchDataset
            Original complete data set.
        idxes : list[int]
            The position of the column to be woe encoded.
        messenger : list[messenger_factory] or messenger_factory
            For active, it should be list[messenger_factory].
            For passive, it should be messenger_factory.

        Attributes
        ----------
        split : dict
            Binning split points for woe-encoded features.
            key(int) : The position of the column to be woe encoded.
            value(list) : The binning split points.
        bin_woe : dict
            The woe value of the woe-encoded feature.
            key(int) : The position of the column to be woe encoded.
            value(list) : The woe values.
        bin_iv : dict
            The iv value of the woe-encoded feature.
            key(int) : The position of the column to be woe encoded.
            value(list) : The iv values.
        """
        self.dataset = dataset
        self.idxes = idxes
        self.split = dict()
        self.bin_woe = dict()
        self.bin_iv = dict()

    def _cal_woe(self, y, role, modify):
        bin = 3
        delta = 1e-07
        features = self.dataset.features
        ids = np.array(self.dataset.ids)
        if isinstance(features, np.ndarray):
            features = features.astype(float)
            positive = np.count_nonzero(y == 1)
            negative = np.count_nonzero(y == 0)
            for i in range(len(self.idxes)):
                bin_feature, self.split[self.idxes[i]] = pd.cut(
                    features[:, self.idxes[i]], bin, labels=False, retbins=True
                )
                self.split[self.idxes[i]] = self.split[self.idxes[i]][1:-1]
                woe = []
                iv = 0
                for j in range(bin):
                    bin_sample = bin_feature == j
                    bin_num = np.count_nonzero(bin_sample)
                    bin_positive = np.count_nonzero(np.logical_and(y == 1, bin_sample))
                    bin_positive_ratio = bin_positive / positive
                    bin_negative_ratio = (bin_num - bin_positive) / negative
                    if bin_positive_ratio == 0 or bin_negative_ratio == 0:
                        bin_woe = np.log(
                            (bin_positive_ratio + delta) / (bin_negative_ratio + delta)
                        )
                    else:
                        bin_woe = np.log(bin_positive_ratio / bin_negative_ratio)
                    bin_iv = (bin_positive_ratio - bin_negative_ratio) * bin_woe
                    bin_feature = np.where(bin_feature == j, bin_woe, bin_feature)
                    woe.append(bin_woe)
                    iv += bin_iv
                self.bin_woe[self.idxes[i]] = woe
                self.bin_iv[self.idxes[i]] = iv
                features[:, self.idxes[i]] = bin_feature
            if role == "active":
                dataset = np.concatenate(
                    (ids[:, np.newaxis], y[:, np.newaxis], features), axis=1
                )
            else:
                dataset = np.concatenate((ids[:, np.newaxis], features), axis=1)
        elif isinstance(features, torch.Tensor):
            features = features.float().numpy()
            y = y.numpy()
            positive = np.count_nonzero(y == 1)
            negative = np.count_nonzero(y == 0)
            for i in range(len(self.idxes)):
                bin_feature, self.split[self.idxes[i]] = pd.cut(
                    features[:, self.idxes[i]], bin, labels=False, retbins=True
                )
                self.split[self.idxes[i]] = self.split[self.idxes[i]][1:-1]
                woe = []
                iv = []
                for j in range(bin):
                    bin_sample = bin_feature == j
                    bin_num = np.count_nonzero(bin_sample)
                    bin_positive = np.count_nonzero(np.logical_and(y == 1, bin_sample))
                    bin_positive_ratio = bin_positive / positive
                    bin_negative_ratio = (bin_num - bin_positive) / negative
                    if bin_positive_ratio == 0 or bin_negative_ratio == 0:
                        bin_woe = np.log(
                            (bin_positive_ratio + delta) / (bin_negative_ratio + delta)
                        )
                    else:
                        bin_woe = np.log(bin_positive_ratio / bin_negative_ratio)
                    bin_iv = (bin_positive_ratio - bin_negative_ratio) * bin_woe
                    bin_feature = np.where(bin_feature == j, bin_woe, bin_feature)
                    woe.append(bin_woe)
                    iv.append(bin_iv)
                self.bin_woe[self.idxes[i]] = woe
                self.bin_iv[self.idxes[i]] = iv
                features[:, self.idxes[i]] = bin_feature
            if role == "active":
                dataset = np.concatenate(
                    (ids[:, np.newaxis], y[:, np.newaxis], features), axis=1
                )
            else:
                dataset = np.concatenate((ids[:, np.newaxis], features), axis=1)
            dataset = torch.from_numpy(dataset)
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )
        if modify:
            self.dataset.set_dataset(dataset)
        return self.split, self.bin_woe, self.bin_iv


class ActiveWoe(Basewoe):
    def __init__(self, dataset, idxes, messenger):
        """Woe encodes the specified features and calculates their iv values for active.

        Parameters
        ----------
        messenger : list[messenger_factory]
        """
        super(ActiveWoe, self).__init__(dataset, idxes)
        self.messenger = messenger

    def cal_woe(self, modify=True):
        y = self.dataset.labels
        for msger in self.messenger:
            msger.send(y)
        super()._cal_woe(y, "active", modify)

        return self.split, self.bin_woe, self.bin_iv


class PassiveWoe(Basewoe):
    def __init__(self, dataset, idxes, messenger):
        """Woe encodes the specified features and calculates their iv values for passive

        Parameters
        ----------
        messenger : list[messenger_factory]
        """
        super(PassiveWoe, self).__init__(dataset, idxes)
        self.messenger = messenger

    def cal_woe(self, modify=True):
        y = self.messenger.recv()
        super()._cal_woe(y, "passive", modify)

        return self.split, self.bin_woe, self.bin_iv


class TestWoe(Basewoe):
    def __init__(self, dataset, idxes, messenger, split, bin_woe):
        super(TestWoe, self).__init__(dataset, idxes)
        self.split = split
        self.bin_woe = bin_woe

    def cal_woe(self):
        features = self.dataset.features
        if isinstance(features, np.ndarray):
            features = features.astype(float)
            sam_num = self.dataset.n_samples
            for idxes_idx in range(len(self.idxes)):
                cur_split = self.split[idxes_idx]
                cur_woe_list = self.bin_woe[idxes_idx]
                for sam_idx in range(sam_num):
                    bin_idx = 0
                    while bin_idx < len(cur_split):
                        if features[sam_idx, idxes_idx] <= cur_split[bin_idx]:
                            break
                        bin_idx += 1
                    self.dataset.features[sam_idx, idxes_idx] = cur_woe_list[bin_idx]
