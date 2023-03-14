import random
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch

from linkefl.base import BaseCryptoSystem, BaseMessenger
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset, TorchDataset


class BasePearsonVfl(ABC):
    def __init__(self, dataset, messenger):
        self.dataset = dataset
        self.messenger = messenger

    @abstractmethod
    def pearson_vfl(self):
        pass


class ActivePearson(BasePearsonVfl):
    def __init__(
        self,
        dataset: Union[NumpyDataset, TorchDataset],
        messenger: List[BaseMessenger],
        cryptosystem: BaseCryptoSystem,
    ):
        super(ActivePearson, self).__init__(dataset, messenger)
        self.cryptosystem = cryptosystem

    def pearson_vfl(self):
        y = self.dataset.labels
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            if isinstance(y, torch.Tensor):
                y = y.numpy()
            meany = y.mean()
            stdy = y.std()
            y_meany = y - meany

            # The active party sends the encrypted vector Y_meanY to the passive party.
            enc_ymeany = np.array(self.cryptosystem.encrypt_vector(y_meany))
            for msger in self.messenger:
                msger.send(enc_ymeany)

            # The active party receives the encrypted cov(X, Y)
            # and stdx_r from the passive party.
            for msger in self.messenger:
                enc_cov = msger.recv()
            for msger in self.messenger:
                stdx_r = msger.recv()

            # The active party calculates the pearson_r
            # and sends it to the passive party.
            cov = np.array(self.cryptosystem.decrypt_vector(enc_cov))
            pearson_r = cov / (stdx_r * stdy)
            for msger in self.messenger:
                msger.send(pearson_r)

        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )

    def pearson_single(self):
        features = self.dataset.features
        y = self.dataset.labels
        n_features = self.dataset.n_features
        peason = []
        for i in range(n_features):
            temp = np.corrcoef(features[:, i], y, rowvar=False)
            peason.append(temp[0][1])
        return peason


class PassivePearson(BasePearsonVfl):
    def pearson_vfl(self):
        x = self.dataset.features
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            if isinstance(x, torch.Tensor):
                x = x.numpy()
            n_samples = self.dataset.n_samples
            n_features = self.dataset.n_features

            meanx = np.empty([n_features])
            x_meanx = np.empty([n_samples, n_features])
            stdx = np.empty([n_features])
            for i in range(n_features):
                meanx[i] = x[:, i].mean()
                x_meanx[:, i] = x[:, i] - meanx[i]
                stdx[i] = x[:, i].std()

            # The passive party receives the enc_y_meany from the active party.
            enc_y_meany = self.messenger.recv()

            # The passive party calculates the stdx_r and enc_cov to the active party.
            r = random.random()
            stdx_r = stdx / r
            enc_cov = []
            for i in range(n_features):
                temp_mul = x_meanx[:, i] * enc_y_meany
                temp_add = 0
                for j in range(n_samples):
                    temp_add += temp_mul[j]
                temp_evc_cov_i = temp_add * (1 / n_samples)
                enc_cov.append(temp_evc_cov_i)
            self.messenger.send(enc_cov)
            self.messenger.send(stdx_r)

            # The passive party receives the pearson_r from the active party.
            pearson_r = self.messenger.recv()

            # The passive party calculates the pearson correlation coefficient.
            pearson = pearson_r / r
            return pearson

        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )
