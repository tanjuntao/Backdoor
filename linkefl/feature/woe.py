from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from linkefl.base import BaseCryptoSystem, BaseMessenger
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset, TorchDataset
from linkefl.dataio.common_dataset import CommonDataset


class Basewoe(ABC):
    def __init__(
        self,
        modify,
        idxes: List[int],
        label=None
    ):
        """Woe encodes the specified features and calculates their iv values.

        Parameters
        ----------
        dataset : NumpyDataset / TorchDataset
            Original complete data set.
        idxes : list[int]
            The position of the column to be woe encoded.

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
            value(float) : The iv values.
        """
        self.modify = modify
        self.idxes = idxes
        self.label = label
        self.split: Dict[int, list] = dict()
        self.bin_woe: Dict[int, list] = dict()
        self.bin_iv: Dict[int, float] = dict()

    def __call__(self, dataset, role):
        bin = 3
        delta = 1e-07
        features = dataset.features
        ids = np.array(dataset.ids)
        if isinstance(features, np.ndarray):
            features = features.astype(float)
            positive = np.count_nonzero(self.label == 1)
            negative = np.count_nonzero(self.label == 0)
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
                    bin_positive = np.count_nonzero(np.logical_and(self.label == 1, bin_sample))
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
            if role == Const.ACTIVE_NAME:
                new_dataset = np.concatenate(
                    (ids[:, np.newaxis], self.label[:, np.newaxis], features), axis=1
                )
            else:
                new_dataset = np.concatenate((ids[:, np.newaxis], features), axis=1)
        elif isinstance(features, torch.Tensor):
            features = features.float().numpy()
            self.label= self.label.numpy()
            positive = np.count_nonzero(self.label == 1)
            negative = np.count_nonzero(self.label == 0)
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
                    bin_positive = np.count_nonzero(np.logical_and(self.label == 1, bin_sample))
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
                new_dataset = np.concatenate(
                    (ids[:, np.newaxis], self.label[:, np.newaxis], features), axis=1
                )
            else:
                new_dataset = np.concatenate((ids[:, np.newaxis], features), axis=1)
            new_dataset = torch.from_numpy(new_dataset)
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )
        if self.modify:
            dataset.set_dataset(new_dataset)
        return dataset


class ActiveWoe(Basewoe):
    def __init__(
        self,
        idxes: List[int],
        modify,
        crypto_type,
        messengers: List[BaseMessenger],
        cryptosystem: Optional[BaseCryptoSystem] = None,
    ):
        """Woe encodes the specified features and calculates their iv values for active.

        Parameters
        ----------
        messengers : list[messenger_factory]
        """
        super(ActiveWoe, self).__init__(modify, idxes)
        self.crypro_type = crypto_type
        self.messengers = messengers
        self.cryptosystem = cryptosystem

    def __call__(self, dataset, role):
        """
        :returns
        woe : list
            The element : dic
            representing the woe values corresponding to the features of a passive party
            performing woe calculation.
                key(int) :  The feature column index for woe,
                value(list) : The corresponding woe value.

        iv : list
            The element : dic
            representing the iv values corresponding to the features of a passive party
            performing iv calculation.
                key(int) :  The feature column index for iv,
                value(float) : The corresponding iv value.
        """
        self.label = dataset.labels
        if self.crypro_type == Const.PLAIN:
            for msger in self.messengers:
                start_signal = msger.recv()  # noqa: F841
                msger.send(self.label)
            super().__call__(dataset, role)
        else:
            enc_y = self.cryptosystem.encrypt_vector(self.label)
            enc_1_y = self.cryptosystem.encrypt_vector(1 - self.label)
            for msger in self.messengers:
                start_signal = msger.recv()
                msger.send(enc_y)
                msger.send(enc_1_y)
            all_enc_sum = []
            for msger in self.messengers:
                all_enc_sum.append(msger.recv())
            self.bin_woe, self.bin_iv = self._cal_woe_encry(dataset, all_enc_sum)
        return dataset

    def _cal_woe_encry(self, dataset, all_enc_sum):
        if isinstance(dataset.features, np.ndarray) or isinstance(
            dataset.features, torch.Tensor
        ):
            if isinstance(dataset.features, torch.Tensor):
                self.label = self.label.numpy()
            ids = np.array(dataset.ids)
            positive = np.count_nonzero(self.label == 1)
            negative = np.count_nonzero(self.label == 0)
            delta = 1e-07
            woe = []
            iv = []
            for i in range(len(all_enc_sum)):
                woe_i = {}
                iv_i = {}
                for idx, enc_sum_bin in all_enc_sum[i].items():
                    sum_bin = self.cryptosystem.decrypt_vector(all_enc_sum[i][idx])
                    woe_i_bin = []
                    iv_i_bin = 0.0
                    for k in range(len(sum_bin)):
                        sum_bin[k] = self.cryptosystem.decrypt_vector(sum_bin[k])
                        bin_positive_ratio = sum_bin[k][0] / positive
                        bin_negative_ratio = sum_bin[k][1] / negative
                        if bin_positive_ratio == 0 or bin_negative_ratio == 0:
                            bin_woe = np.log(
                                (bin_positive_ratio + delta)
                                / (bin_negative_ratio + delta)
                            )
                        else:
                            bin_woe = np.log(bin_positive_ratio / bin_negative_ratio)
                        bin_iv = (bin_positive_ratio - bin_negative_ratio) * bin_woe
                        woe_i_bin.append(bin_woe)
                        iv_i_bin += bin_iv
                    woe_i[idx] = woe_i_bin
                    iv_i[idx] = iv_i_bin
                woe.append(woe_i)
                iv.append(iv_i)
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )
        return woe, iv


class PassiveWoe(Basewoe):
    def __init__(
        self,
        idxes: List[int],
        modify,
        crypto_type,
        messenger: BaseMessenger,
        cryptosystem: Optional[BaseCryptoSystem] = None,
    ):
        """Woe encodes the specified features
        and calculates their iv values for passive.

        Parameters
        ----------
        """
        super(PassiveWoe, self).__init__(modify, idxes)
        self.crypto_type = crypto_type
        self.messenger = messenger
        self.cryptosystem = cryptosystem

    def __call__(self, dataset, role):
        if self.crypto_type == Const.PLAIN:
            self.messenger.send(Const.START_SIGNAL)
            self.label = self.messenger.recv()
            super().__call__(dataset, role)
        else:
            self.messenger.send(Const.START_SIGNAL)
            enc_y = self.messenger.recv()
            enc_1_y = self.messenger.recv()
            enc_sum = self._cal_woe_encry(dataset, enc_y, enc_1_y)
            self.messenger.send(enc_sum)
        return dataset

    def _cal_woe_encry(self, dataset, enc_y, enc_1_y):
        bin = 3
        features = dataset.features
        n_samples = dataset.n_samples
        enc_sum = {}
        if isinstance(features, np.ndarray) or isinstance(features, torch.Tensor):
            if isinstance(features, torch.Tensor):
                features = features.float().numpy()
            for i in range(len(self.idxes)):
                bin_feature, self.split[self.idxes[i]] = pd.cut(
                    features[:, self.idxes[i]], bin, labels=False, retbins=True
                )
                self.split[self.idxes[i]] = self.split[self.idxes[i]][1:-1]
                enc_bin_sum = []
                for j in range(bin):
                    bin_sample = bin_feature == j
                    enc_bin_sum_y = 0
                    enc_bin_sum_1_y = 0
                    for k in range(n_samples):
                        if bin_sample[k]:
                            enc_bin_sum_y += enc_y[k]
                            enc_bin_sum_1_y += enc_1_y[k]
                    enc_bin_sum.append([enc_bin_sum_y, enc_bin_sum_1_y])
                enc_sum[self.idxes[i]] = enc_bin_sum
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )
        return enc_sum


class TestWoe:
    def __init__(self, dataset, woe_features, messenger, split, bin_woe):
        self.split = split
        self.bin_woe = bin_woe
        self.messenger = messenger
        self.dataset = dataset
        self.woe_features = woe_features

    def cal_woe(self):
        features = self.dataset.features
        if isinstance(features, np.ndarray):
            features = features.astype(float)
            sam_num = self.dataset.n_samples
            for woe_features_idx in range(len(self.woe_features)):
                cur_split = self.split[woe_features_idx]
                cur_woe_list = self.bin_woe[woe_features_idx]
                for sam_idx in range(sam_num):
                    bin_idx = 0
                    while bin_idx < len(cur_split):
                        if features[sam_idx, woe_features_idx] <= cur_split[bin_idx]:
                            break
                        bin_idx += 1
                    self.dataset.features[sam_idx, woe_features_idx] = cur_woe_list[
                        bin_idx
                    ]
