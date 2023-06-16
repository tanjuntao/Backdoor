#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : LinkeFL
# @File    : sample.py
# @Author  : HaoRan Cheng
# @Date    : 2023/5/9 16:59

import random
import numpy as np


class Sample:
    @staticmethod
    def random_sampling(grad, hess, sample_rate):
        """
        Sample-level random sampling.

        Args:
            grad: np.array
            hess: np.array
            sample_rate: float

        Returns:
            list, [select_grad, select_hess, select_idx.]
        """
        sample_num = grad.shape[0]
        selected_idx = random.sample(
            list(range(sample_num)), int(sample_num * sample_rate)
        )
        selected_idx.sort()

        selected_g, selected_h = grad[selected_idx], grad[selected_idx]
        return [selected_g, selected_h, selected_idx]

    @staticmethod
    def goss_sampling(grad, hess, top_rate, other_rate):
        """
        Sample-level sampling method proposed in lightGBM.

        Args:
            grad: np.array
            hess: np.array

        Returns:
            list, [select_grad, select_hess, select_idx.]
        """
        # if it is multi-classification case, we need to sum g
        if len(grad.shape) > 1:
            abs_g_sum_arr = np.abs(grad).sum(axis=1)
        else:
            abs_g_sum_arr = np.abs(grad)

        # abs_g_list_arr = g_sum_arr
        sorted_idx = np.argsort(-abs_g_sum_arr, kind="stable")  # stable sample result

        sample_num = len(abs_g_sum_arr)
        a_part_num = int(sample_num * top_rate)
        b_part_num = int(sample_num * other_rate)

        if a_part_num == 0 or b_part_num == 0:
            raise ValueError(
                "subsampled result is 0: top sample {}, other sample {}".format(
                    a_part_num, b_part_num
                )
            )

        # index of a part
        a_sample_idx = sorted_idx[:a_part_num]

        # index of b part
        rest_sample_idx = sorted_idx[a_part_num:]
        b_sample_idx = np.random.choice(rest_sample_idx, size=b_part_num, replace=False)

        # small gradient sample weights
        amplify_weights = (1 - top_rate) / other_rate

        grad[b_sample_idx] *= amplify_weights
        hess[b_sample_idx] *= amplify_weights

        # get selected sample
        a_idx_set, b_idx_set = set(list(a_sample_idx)), set(list(b_sample_idx))
        idx_set = a_idx_set.union(b_idx_set)
        selected_idx = np.array(list(idx_set))

        selected_g, selected_h = grad[selected_idx], hess[selected_idx]

        return selected_g, selected_h, selected_idx
