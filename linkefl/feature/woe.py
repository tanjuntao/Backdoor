from abc import ABC, abstractmethod

import numpy as np
import torch
import pandas as pd


class Basewoe(ABC):
    def __init__(self, dataset, woe_features, messenger):
        self.dataset = dataset
        self.woe_features = woe_features
        self.messenger = messenger
        self.split = dict()
        self.bin_woe = dict()

    @abstractmethod
    def cal_woe(self):
        pass


class ActiveWoe(Basewoe):
    def __init__(self, dataset, woe_features, messenger):
        super(ActiveWoe, self).__init__(dataset, woe_features, messenger)

    def cal_woe(self):
        bin = 3
        delta = 1e-07
        features = self.dataset.features
        y = self.dataset.labels
        for msger in self.messenger: msger.send(y)
        ids = np.array(self.dataset.ids)
        if isinstance(features, np.ndarray):
            features = features.astype(float)
            positive = np.count_nonzero(y == 1)
            negative = np.count_nonzero(y == 0)
            for i in range(len(self.woe_features)):
                bin_feature, self.split[self.woe_features[i]] = \
                    pd.cut(features[:, self.woe_features[i]], bin, labels=False, retbins=True)
                self.split[self.woe_features[i]] = self.split[self.woe_features[i]][1: -1]
                woe = []
                for j in range(bin):
                    bin_sample = (bin_feature == j)
                    bin_num = np.count_nonzero(bin_sample)
                    bin_positive = np.count_nonzero(np.logical_and(y == 1, bin_sample))
                    bin_positive_ratio = bin_positive / positive
                    bin_negative_ratio = (bin_num - bin_positive) / negative
                    if bin_positive_ratio == 0 or bin_negative_ratio == 0:
                        bin_woe = np.log((bin_positive_ratio + delta) / (bin_negative_ratio + delta))
                    else:
                        bin_woe = np.log(bin_positive_ratio / bin_negative_ratio)
                    bin_feature = \
                        np.where(bin_feature == j, bin_woe, bin_feature)
                    woe.append(bin_woe)
                self.bin_woe[self.woe_features[i]] = woe
                features[:, self.woe_features[i]] = bin_feature
            dataset = np.concatenate(
                (ids[:, np.newaxis], y[:, np.newaxis], features),
                axis=1
            )
        elif isinstance(features, torch.Tensor):
            features = features.float().numpy()
            y = y.numpy()
            positive = np.count_nonzero(y == 1)
            negative = np.count_nonzero(y == 0)
            for i in range(len(self.woe_features)):
                bin_feature, self.split[self.woe_features[i]] = \
                    pd.cut(features[:, self.woe_features[i]], bin, labels=False, retbins=True)
                self.split[self.woe_features[i]] = self.split[self.woe_features[i]][1: -1]
                woe = []
                for j in range(bin):
                    bin_sample = (bin_feature == j)
                    bin_num = np.count_nonzero(bin_sample)
                    bin_positive = np.count_nonzero(np.logical_and(y == 1, bin_sample))
                    bin_positive_ratio = bin_positive / positive
                    bin_negative_ratio = (bin_num - bin_positive) / negative
                    if bin_positive_ratio == 0 or bin_negative_ratio == 0:
                        bin_woe = np.log((bin_positive_ratio + delta) / (bin_negative_ratio + delta))
                    else:
                        bin_woe = np.log(bin_positive_ratio / bin_negative_ratio)
                    bin_feature = \
                        np.where(bin_feature == j, bin_woe, bin_feature)
                    woe.append(bin_woe)
                self.bin_woe[self.woe_features[i]] = woe
                features[:, self.woe_features[i]] = bin_feature
            dataset = np.concatenate(
                (ids[:, np.newaxis], y[:, np.newaxis], features),
                axis=1
            )
            dataset = torch.from_numpy(dataset)
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')
        self.dataset.set_dataset(dataset)

        return self.split, self.bin_woe


class PassiveWoe(Basewoe):
    def __init__(self, dataset, woe_features, messenger):
        super(PassiveWoe, self).__init__(dataset, woe_features, messenger)

    def cal_woe(self):
        bin = 3
        delta = 1e-07
        features = self.dataset.features
        y = self.messenger.recv()
        ids = np.array(self.dataset.ids)
        if isinstance(features, np.ndarray):
            features = features.astype(float)
            positive = np.count_nonzero(y == 1)
            negative = np.count_nonzero(y == 0)
            for i in range(len(self.woe_features)):
                bin_feature, self.split[self.woe_features[i]] = \
                    pd.cut(features[:, self.woe_features[i]], bin, labels=False, retbins=True)
                self.split[self.woe_features[i]] = self.split[self.woe_features[i]][1: -1]
                woe = []
                for j in range(bin):
                    bin_sample = (bin_feature == j)
                    bin_num = np.count_nonzero(bin_sample)
                    bin_positive = np.count_nonzero(np.logical_and(y == 1, bin_sample))
                    bin_positive_ratio = bin_positive / positive
                    bin_negative_ratio = (bin_num - bin_positive) / negative
                    if bin_positive_ratio == 0 or bin_negative_ratio == 0:
                        bin_woe = np.log((bin_positive_ratio + delta) / (bin_negative_ratio + delta))
                    else:
                        bin_woe = np.log(bin_positive_ratio / bin_negative_ratio)
                    bin_feature = \
                        np.where(bin_feature == j, bin_woe, bin_feature)
                    woe.append(bin_woe)
                self.bin_woe[self.woe_features[i]] = woe
                features[:, self.woe_features[i]] = bin_feature
            dataset = np.concatenate(
                (ids[:, np.newaxis], features),
                axis=1
            )
        elif isinstance(features, torch.Tensor):
            features = features.float().numpy()
            y = y.numpy()
            positive = np.count_nonzero(y == 1)
            negative = np.count_nonzero(y == 0)
            for i in range(len(self.woe_features)):
                bin_feature, self.split[self.woe_features[i]] = \
                    pd.cut(features[:, self.woe_features[i]], bin, labels=False, retbins=True)
                self.split[self.woe_features[i]] = self.split[self.woe_features[i]][1: -1]
                woe = []
                for j in range(bin):
                    bin_sample = (bin_feature == j)
                    bin_num = np.count_nonzero(bin_sample)
                    bin_positive = np.count_nonzero(np.logical_and(y == 1, bin_sample))
                    bin_positive_ratio = bin_positive / positive
                    bin_negative_ratio = (bin_num - bin_positive) / negative
                    if bin_positive_ratio == 0 or bin_negative_ratio == 0:
                        bin_woe = np.log((bin_positive_ratio + delta) / (bin_negative_ratio + delta))
                    else:
                        bin_woe = np.log(bin_positive_ratio / bin_negative_ratio)
                    bin_feature = \
                        np.where(bin_feature == j, bin_woe, bin_feature)
                    woe.append(bin_woe)
                self.bin_woe[self.woe_features[i]] = woe
                features[:, self.woe_features[i]] = bin_feature
            dataset = np.concatenate(
                (ids[:, np.newaxis], features),
                axis=1
            )
            dataset = torch.from_numpy(dataset)
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')
        self.dataset.set_dataset(dataset)

        return self.split, self.bin_woe
