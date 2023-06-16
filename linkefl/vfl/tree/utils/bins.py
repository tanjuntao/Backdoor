#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : LinkeFL
# @File    : bins.py
# @Author  : HaoRan Cheng
# @Date    : 2023/5/26 17:10

import numpy as np


class Bins:
    @classmethod
    def get_bin_info(cls, x_train, max_bin, pool):
        """compute hist information

        Args:
            x_train: training data，size = sample * feature
            max_bin: max bin number for a feature point

        Returns:
            bin_index: bin index of each feature point in the
                complete feature hist (a column)，size = sample * feature
            bin_split: split point in the
                complete feature hist (a column)，size = feature * bin_of_this_feature
        """

        bin_index = np.empty_like(x_train, dtype=int)
        bin_split = [None] * x_train.shape[1]

        if pool is not None:
            threads = []
            for i in range(x_train.shape[1]):
                # find hist for each feature point
                t = pool.apply_async(cls._find_bin, (x_train[:, i], max_bin))
                threads.append(t)
            for i, t in enumerate(threads):
                bin_index[:, i], bin_split[i] = t.get()
        else:
            for i in range(x_train.shape[1]):
                # find hist for each feature point
                bin_index[:, i], bin_split[i] = cls._find_bin(x_train[:, i], max_bin)

        # fill the empty place in bin_split to solve the influence of different
        # bin_num (seems no influence right now)
        # bin_split = np.array(list(itertools.zip_longest(*bin_split, fillvalue=None))).T

        return bin_index, bin_split

    @staticmethod
    def _find_bin(vec, max_bin):
        shape = vec.shape
        vec = vec.flatten()

        distinct_vec, counts = np.unique(vec, return_counts=True)
        sorted_vec = np.sort(vec)

        bin_data_rate = 1 / max_bin
        bin_last_pos = [
            sorted_vec[int((i + 1) * bin_data_rate * len(vec)) - 1]
            for i in range(max_bin)
        ]
        vec_bin_end = np.unique(bin_last_pos)

        vec_bin_index = np.empty_like(vec, dtype=int)
        index = 0
        for data, count in zip(distinct_vec, counts):
            if data > vec_bin_end[index]:
                index += 1
            vec_bin_index[vec == data] = index
        vec_bin_index = vec_bin_index.reshape(shape)

        return vec_bin_index, vec_bin_end[:-1]
