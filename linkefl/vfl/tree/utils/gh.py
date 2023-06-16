#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : LinkeFL
# @File    : gh.py
# @Author  : HaoRan Cheng
# @Date    : 2023/5/9 20:54

import numpy as np


class GH:
    @staticmethod
    def gh_packing(grad, hess, r):
        g_int_max = round(len(grad) * np.abs(grad).max() * (1 << r))
        h_int_max = round(len(hess) * np.abs(hess).max() * (1 << r))

        g_length = g_int_max.bit_length()
        h_length = h_int_max.bit_length()
        gh_length = (g_length + 1) + h_length

        # transform g, h from float to big int
        grad_int = np.floor(grad * (1 << 64)) >> (64 - r)
        hess_int = np.floor(hess * (1 << 64)) >> (64 - r)

        # pack g, h
        gh_int = (grad_int << h_length) + hess_int

        return gh_int, h_length, gh_length

    @classmethod
    def gh_compress_multi(cls, grad, hess, r, key_size):
        gh_int, h_length, gh_length = cls.gh_packing(grad, hess, r)

        plaintext_bit = key_size - 1
        capacity = int(plaintext_bit / gh_length)  # max number of gh in a cipher space

        gh_compress = []

        for sample_id in range(grad.shape[0]):
            gh_compress_vec, big_num, count = [], 0, 0

            for label_id in range(grad.shape[1]):
                gh = gh_int[sample_id][label_id]
                big_num = (big_num << gh_length) + gh
                count += 1

                if count == capacity:
                    gh_compress_vec.append(big_num)
                    big_num, count = 0, 0

            if count != 0:
                gh_compress_vec.append(big_num)

            gh_compress.append(gh_compress_vec)

        return np.array(gh_compress), h_length, gh_length
