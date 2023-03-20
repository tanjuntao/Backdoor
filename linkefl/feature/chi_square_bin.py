from abc import ABC
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch

from linkefl.base import BaseMessenger
from linkefl.dataio import NumpyDataset, TorchDataset


class BaseChiBin(ABC):
    def __init__(self, dataset, idxes, max_group):
        """Chi-square binning for specified features.

        Parameters
        ----------
        dataset : NumpyDataset / TorchDataset
            Original complete data set.
        idxes : list[int]
            The position of the column to be woe encoded.
        max_group : int
            The maximum number of bins.

        Attributes
        ----------
        bin : dict
            The binning results of the binned features
            key(int) : The position of the column to be woe encoded.
            value(list) : The binning results.
        """
        self.dataset = dataset
        self.idxes = idxes
        self.max_group = max_group
        self.bin: Dict[int, list] = dict()

    def _chi(self, arr):
        """Calculate chi-square value.

        Parameters
        ----------
        arr : 2D ndarray.
            Frequency statistics table.
        """
        assert arr.ndim == 2
        # Calculate total frequency for each row.
        R_N = arr.sum(axis=1)
        # total frequency per column
        C_N = arr.sum(axis=0)
        # total frequency
        N = arr.sum()
        # Calculate expected frequency. C_i * R_j / N。
        E = np.ones(arr.shape) * C_N / N
        E = (E.T * R_N).T
        square = (arr - E) ** 2 / E
        # When the expected frequency is 0,
        # it is meaningless to do the divisor,
        # and it will not be included in the chi-square value.
        square[E == 0] = 0
        # chi-square value
        v = square.sum()
        return v

    def _chiMerge(self, y):
        """Find the cutoff point of chi-square binning.

        Parameters
        ----------
        y：ndarry
        Return: The dividing point after each feature binning.
        """
        cutoffs = dict()
        features = self.dataset.features
        for i in self.idxes:
            freq_tab = pd.crosstab(features[:, i], y)
            # 转成numpy数组用于计算。
            freq = freq_tab.values
            # 初始分组切分点，每个变量值都是切分点。每组中只包含一个变量值.
            # 分组区间是左闭右开的，如cutoffs = [1,2,3]，则表示区间 [1,2) , [2,3) ,[3,3+)。
            bin_cutoffs = freq_tab.index.values

            while True:
                minvalue = None
                minidx = None
                # 从第1组开始，依次取两组计算卡方值，并判断是否小于当前最小的卡方
                for j in range(len(freq) - 1):
                    v = self._chi(freq[j : j + 2])
                    if minvalue is None or (minvalue > v):  # 小于当前最小卡方，更新最小值
                        minvalue = v
                        minidx = j

                # 如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
                if self.max_group < len(freq):
                    # minidx后一行合并到minidx
                    tmp = freq[minidx] + freq[minidx + 1]
                    freq[minidx] = tmp
                    # 删除minidx后一行
                    freq = np.delete(freq, minidx + 1, 0)
                    # 删除对应的切分点
                    bin_cutoffs = np.delete(bin_cutoffs, minidx + 1, 0)

                else:  # 最小卡方值不小于阈值，停止合并。
                    break
            cutoffs[i] = bin_cutoffs
        return cutoffs

    def _value2group(self, x, cutoffs):
        """Transform the value of a variable into the corresponding group.

        Parameters
        ----------
        x : int/float
            The value that needs to be converted to the group.
        cutoffs : ndarray
            Cut-off point for each group.
        ----------
        Return : The group corresponding to x, such as group1. Start with group1.
        """
        # Cut-off points are sorted from small to large.
        cutoffs = sorted(cutoffs)
        num_groups = len(cutoffs)

        # Exception: less than the starting value of the first group.
        # Put it directly in the first group here.
        if x < cutoffs[0]:
            return "group1"

        for i in range(1, num_groups):
            if cutoffs[i - 1] <= x < cutoffs[i]:
                return "group{}".format(i)

        # The last group, and it may also include some very large outliers.
        return "group{}".format(num_groups)

    def _chi_bin(self, y):
        cutoffs = self._chiMerge(y)
        features = self.dataset.features
        if isinstance(features, np.ndarray):
            data = pd.DataFrame(features)
            for i in self.idxes:
                self.bin[i] = (
                    data.iloc[:, i].apply(self._value2group, args=(cutoffs[i],)).values
                )
        elif isinstance(features, torch.Tensor):
            data = features.numpy()
            data = pd.DataFrame(data)
            for i in self.idxes:
                self.bin[i] = (
                    data.iloc[:, i].apply(self._value2group, args=(cutoffs[i],)).values
                )
        else:
            raise TypeError(
                "dataset should be an instance of numpy.ndarray or torch.Tensor"
            )


class ActiveChiBin(BaseChiBin):
    def __init__(
        self,
        dataset: Union[NumpyDataset, TorchDataset],
        idxes: List[int],
        messengers: List[BaseMessenger],
        max_group: int = 5,
    ):
        """Find the cutoff point of chi-square binning for active.

        Parameters
        ----------
        messengers : list[messenger_factory]
        """
        super(ActiveChiBin, self).__init__(dataset, idxes, max_group)
        self.messengers = messengers

    def chi_bin(self) -> Dict[int, list]:
        y = self.dataset.labels
        for msger in self.messengers:
            msger.send(y)
        super()._chi_bin(y)

        return self.bin


class PassiveChiBin(BaseChiBin):
    def __init__(
        self,
        dataset: Union[NumpyDataset, TorchDataset],
        idxes: List[int],
        messenger: BaseMessenger,
        max_group: int = 5,
    ):
        """Find the cutoff point of chi-square binning for passive.

        Parameters
        ----------
        messenger : BaseMessenger
        """
        super(PassiveChiBin, self).__init__(dataset, idxes, max_group)
        self.messenger = messenger

    def chi_bin(self) -> Dict[int, list]:
        y = self.messenger.recv()
        super()._chi_bin(y)

        return self.bin
