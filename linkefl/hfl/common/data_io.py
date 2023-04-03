import io
import os
import random
from typing import Union
from urllib.error import URLError

import numpy as np
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from linkefl.dataio import NumpyDataset
from linkefl.util import urlretrive


class MyData(Dataset):
    def __init__(self, name, root, train, download):
        def _check_exists(dataset_name, root_, train_, resources_):
            if train_:
                filename_ = resources_[dataset_name][0]
            else:
                filename_ = resources_[dataset_name][1]
            return os.path.exists(os.path.join(root_, filename_))

        resources = {
            "cancer": ("cancer-train.csv", "cancer-test.csv"),
            "digits": ("digits-train.csv", "digits-test.csv"),
            "diabetes": ("diabetes-train.csv", "diabetes-test.csv"),
            "iris": ("iris-train.csv", "iris-test.csv"),
            "wine": ("wine-train.csv", "wine-test.csv"),
            "epsilon": ("epsilon-train.csv", "epsilon-test.csv"),
            "census": ("census-train.csv", "census-test.csv"),
            "credit": ("credit-train.csv", "credit-test.csv"),
            "default_credit": ("default-credit-train.csv", "default-credit-test.csv"),
            "covertype": ("covertype-train.csv", "covertype-test.csv"),
            "criteo": ("criteo-train.csv", "criteo-test.csv"),
            "higgs": ("higgs-train.csv", "higgs-test.csv"),
            "year": ("year-train.csv", "year-test.csv"),
            "nyc_taxi": ("nyc-taxi-train.csv", "nyc-taxi-test.csv"),
            "avazu": ("avazu-train.csv", "avazu-test.csv"),
        }
        BASE_URL = "https://linkefl.cyh.me/d/linkefl/datasets/"
        root = os.path.join(root, "tabular")

        if download:
            if _check_exists(name, root, train, resources):
                # if data files have already been downloaded, then skip this branch
                print("Data files have already been downloaded.")
            else:
                # download data files from web server
                os.makedirs(root, exist_ok=True)
                filename = resources[name][0] if train else resources[name][1]
                fpath = os.path.join(root, filename)
                full_url = BASE_URL + filename
                try:
                    print("Downloading {} to {}".format(full_url, fpath))
                    urlretrive(full_url, fpath)
                except URLError as error:
                    raise RuntimeError(
                        "Failed to download {} with error message: {}".format(
                            full_url, error
                        )
                    )
                print("Done!")
        if not _check_exists(name, root, train, resources):
            raise RuntimeError(
                "Dataset not found. You can use download=True to get it."
            )

        # ===== 1. Load dataset =====
        if train:
            np_csv = np.genfromtxt(
                os.path.join(root, resources[name][0]), delimiter=",", encoding="utf-8"
            )
        else:
            np_csv = np.genfromtxt(
                os.path.join(root, resources[name][1]), delimiter=",", encoding="utf-8"
            )
        self._ids = np_csv[:, 0]  # no need to convert to integers here
        self._labels = np_csv[:, 1]  # no need to convert to integers here
        self._feats = np_csv[:, 2:]


    def __getitem__(self, index):
        """
        # 已知样本index,返回样本的值（样本和样本label）。
        # 通常是根据读取一个存放了 样本路径和标签信息的txt文档获得具体的数据。
        :param index: 样本的index
        :return: 样本，label
        """
        return self._feats[index], self._labels[index]

    def __len__(self):
        return len(self._feats)

class MyData_tabular(Dataset):
    def __init__(self, np_csv):

        # ===== 1. Load dataset =====
        self._ids = np_csv[:, 0]  # no need to convert to integers here
        self._labels = np_csv[:, 1]  # no need to convert to integers here
        self._feats = np_csv[:, 2:]

    def __getitem__(self, index):
        """
        # 已知样本index,返回样本的值（样本和样本label）。
        # 通常是根据读取一个存放了 样本路径和标签信息的txt文档获得具体的数据。
        :param index: 样本的index
        :return: 样本，label
        """
        return self._feats[index], self._labels[index]

    def __len__(self):
        return len(self._feats)
    def n_features(self):
        return self._feats.shape[1]


def MyData_image(dataset_name,data_path="../../data",train=True,download=True):

    if dataset_name =="mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset = datasets.MNIST(
            data_path, train=train, download=True, transform=transform
        )

    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        dataset = datasets.FashionMNIST(
            data_path, train=train, download=True, transform=transform
        )

    elif dataset_name =="cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10(
            data_path, train=train, download=True, transform=transform
        )

    return dataset
