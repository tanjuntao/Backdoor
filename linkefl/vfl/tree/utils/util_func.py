#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : LinkeFL
# @File    : util_func.py
# @Author  : HaoRan Cheng
# @Date    : 2023/5/9 20:18

import os
import numpy as np


def wrap_message(name, *, content):
    return {"name": name, "content": content}


def one_hot(labels, n_labels):
    labels_onehot = np.zeros((len(labels), n_labels))
    labels_onehot[np.arange(len(labels)), labels] = 1
    return labels_onehot


def get_latest_filename(filedir):
    if os.path.exists(filedir):
        file_list = os.listdir(filedir)
    else:
        raise ValueError("not exist filedir.")

    # sort by create time
    file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(filedir, fn)))
    return file_list[-1]
