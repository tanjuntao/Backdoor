#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : LinkeFL
# @File    : node_cal.py
# @Author  : HaoRan Cheng
# @Date    : 2023/5/9 20:56

import numpy as np


class LeafValue:
    @staticmethod
    def leaf_weight(gh, sample_tag, reg_lambda):
        g_sum, h_sum = np.dot(sample_tag, gh)
        weight = -(g_sum / (h_sum + reg_lambda))

        return weight

    @staticmethod
    def leaf_weight_multi(gh, sample_tag, reg_lambda):
        grad = gh[:, :, 0]
        hess = gh[:, :, 1]
        sample_id = [i for i, tag in enumerate(sample_tag) if tag == 1]
        g_sum = grad[sample_id].sum(axis=0)
        h_sum = hess[sample_id].sum(axis=0)

        weight = -(g_sum / (h_sum + reg_lambda))

        return weight
