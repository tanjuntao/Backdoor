from abc import ABC, abstractmethod

import numpy as np
import torch
import pandas as pd


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
            value(float) : The iv values.
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
            dataset = torch.from_numpy(dataset)
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )
        if modify:
            self.dataset.set_dataset(dataset)
        return self.split, self.bin_woe, self.bin_iv


class ActiveWoe(Basewoe):
    def __init__(self, dataset, idxes, messenger, cryptosystem=None):
        """Woe encodes the specified features and calculates their iv values for active.

        Parameters
        ----------
        messenger : list[messenger_factory]
        """
        super(ActiveWoe, self).__init__(dataset, idxes)
        self.messenger = messenger
        self.cryptosystem = cryptosystem

    def cal_woe(self, modify=True):
        y = self.dataset.labels
        for msger in self.messenger: msger.send(y)
        super()._cal_woe(y, 'active', modify)

        return self.split, self.bin_woe, self.bin_iv

    def cal_woe_encry(self):
        """
        :returns
        woe : list
            The element : dic
            representing the woe values corresponding to the features of a passive party performing woe calculation.
                key(int) :  The feature column index for woe,
                value(list) : The corresponding woe value.

        iv : list
            The element : dic
            representing the iv values corresponding to the features of a passive party performing iv calculation.
                key(int) :  The feature column index for iv,
                value(float) : The corresponding iv value.
        """
        y = self.dataset.labels

        enc_y = self.cryptosystem.encrypt_vector(y)
        enc_1_y = self.cryptosystem.encrypt_vector(1-y)
        for msger in self.messenger:
            msger.send(enc_y)
            msger.send(enc_1_y)

        all_enc_sum = []
        for msger in self.messenger:
            all_enc_sum.append(msger.recv())
        woe, iv = self._cal_woe_encry(all_enc_sum)
        return woe, iv

    def _cal_woe_encry(self, all_enc_sum):
        y = self.dataset.labels
        if isinstance(self.dataset.features, np.ndarray) or isinstance(self.dataset.features, torch.Tensor):
            if isinstance(self.dataset.features, torch.Tensor):
                y = y.numpy()
            positive = np.count_nonzero(y == 1)
            negative = np.count_nonzero(y == 0)
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
                            bin_woe = np.log((bin_positive_ratio + delta) / (bin_negative_ratio + delta))
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
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')
        return woe, iv


class PassiveWoe(Basewoe):
    def __init__(self, dataset, idxes, messenger, cryptosystem=None):
        """Woe encodes the specified features and calculates their iv values for passive.

        Parameters
        ----------
        messenger : list[messenger_factory]
        """
        super(PassiveWoe, self).__init__(dataset, idxes)
        self.messenger = messenger
        self.cryptosystem = cryptosystem

    def cal_woe(self, modify=True):
        y = self.messenger.recv()
        super()._cal_woe(y, "passive", modify)

        return self.split, self.bin_woe, self.bin_iv

    def cal_woe_encry(self):
        enc_y = self.messenger.recv()
        enc_1_y = self.messenger.recv()
        enc_sum = self._cal_woe_encry(enc_y, enc_1_y)
        self.messenger.send(enc_sum)

    def _cal_woe_encry(self, enc_y, enc_1_y):
        bin = 3
        features = self.dataset.features
        n_samples = self.dataset.n_samples
        enc_sum = {}
        if isinstance(features, np.ndarray) or isinstance(features, torch.Tensor):
            if isinstance(features, torch.Tensor):
                features = features.float().numpy()
            for i in range(len(self.idxes)):
                bin_feature, self.split[self.idxes[i]] = \
                    pd.cut(features[:, self.idxes[i]], bin, labels=False, retbins=True)
                self.split[self.idxes[i]] = self.split[self.idxes[i]][1: -1]
                enc_bin_sum = []
                for j in range(bin):
                    bin_sample = (bin_feature == j)
                    enc_bin_sum_y = 0
                    enc_bin_sum_1_y = 0
                    for k in range(n_samples):
                        if bin_sample[k]:
                            enc_bin_sum_y += enc_y[k]
                            enc_bin_sum_1_y += enc_1_y[k]
                    enc_bin_sum.append([enc_bin_sum_y, enc_bin_sum_1_y])
                enc_sum[self.idxes[i]] = enc_bin_sum
        else:
            raise TypeError('dataset should be an instance of numpy.ndarray or torch.Tensor')
        return enc_sum

    
class TestWoe(Basewoe):
    def __init__(self, dataset, idxes, messenger, split, bin_woe):
        super(TestWoe, self).__init__(dataset, idxes)
        self.split = split
        self.bin_woe = bin_woe

    def cal_woe(self):
        features = self.dataset.features
        ids = np.array(self.dataset.ids)
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

def plot_iv(iv_dict, file_dir="./models"):
    '''
    _, _, iv_dict = woe.cal_woe()
    woe is in [ActiveWoe(), PassiveWoe()]
    Then iv_dict is the input of this function
    '''
    import matplotlib.pyplot as plt
    feature_list = list(iv_dict.keys())
    iv_list = [iv_dict[key] for key in feature_list]
    fig, ax = plt.subplots()
    ax.barh(feature_list, iv_list)
    ax.set_xlabel('IV values')
    ax.set_ylabel('Feature Ids')
    ax.set_title('Feature IV Analysis')
    plt.savefig(f"{file_dir}/iv_analysis.png")
    plt.close()