from __future__ import annotations  # python >= 3.7, give type hint before definition

import random
from typing import Union

import numpy as np
import pandas as pd
import torch

from linkefl.common.const import Const
from linkefl.dataio import BaseDataset
# avoid circular importing
# from linkefl.feature.transform import BaseTransform


class CommonDataset(BaseDataset):
    """Common Dataset"""
    def __init__(self,
                 role: str,
                 raw_dataset: Union[np.ndarray, torch.Tensor],
                 dataset_type: str,
                 # transform: BaseTransform = None
                 transform=None
    ):
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), "Invalid role"
        self.role = role
        self.dataset_type = dataset_type
        self.has_label = True if role == Const.ACTIVE_NAME else False

        if transform is not None:
            raw_dataset = transform.fit(raw_dataset, role=role)

        # after this function call, self._raw_dataset attribute will be created
        self.set_dataset(raw_dataset)

    @classmethod
    def train_test_split(cls,
                         dataset: CommonDataset,
                         test_size: float,
                         option: str = Const.SEQUENCE,
                         seed=None
    ):
        """Split the whole dataset into trainset and testset"""
        assert isinstance(dataset, CommonDataset), \
            "dataset should be an instance of CommonDataset"
        assert 0 <= test_size < 1, "test size should be in range [0, 1)"

        if option == Const.SEQUENCE:
            perm = list(range(dataset.n_samples))
        elif option == Const.RANDOM:
            if seed is not None:
                random.seed(seed)
            perm = list(range(dataset.n_samples))
            random.shuffle(perm)
        else:
            raise ValueError(
                "in train_test_split method, the option of sample "
                "permutation can only take from SEQUENCE AND RANDOM."
            )

        n_train_samples = int(dataset.n_samples * (1 - test_size))
        raw_trainset = dataset.get_dataset()[perm[:n_train_samples], :]
        raw_testset = dataset.get_dataset()[perm[n_train_samples:], :]

        trainset = cls(role=dataset.role,
                       raw_dataset=raw_trainset,
                       dataset_type=dataset.dataset_type)
        testset = cls(role=dataset.role,
                      raw_dataset=raw_testset,
                      dataset_type=dataset.dataset_type)

        return trainset, testset

    @classmethod
    def feature_split(cls,
                      dataset: CommonDataset,
                      n_splits: int,
                      option: str = Const.SEQUENCE,
                      seed=None
    ):
        assert isinstance(dataset, CommonDataset), \
            "dataset should be an instance of CommonDataset"
        if option == Const.SEQUENCE:
            perm = list(range(dataset.n_features))
        elif option == Const.RANDOM:
            if seed is not None:
                random.seed(seed)
            perm = list(range(dataset.n_features))
            random.shuffle(perm)
        else:
            raise ValueError(
                "in feature_split method, the option of feature "
                "permutation can only take from SEQUENCE AND RANDOM."
            )

        permed_features = dataset.features[:, perm]
        ids = dataset.ids # is a Python list
        step = dataset.n_features // n_splits

        splitted_datasets = []
        for i in range(n_splits):
            begin_idx = i * step
            if i != n_splits - 1:
                end_idx = (i + 1) * step
            else:
                end_idx = dataset.n_features

            splitted_feats = permed_features[:, begin_idx:end_idx]
            if isinstance(splitted_feats, np.ndarray): # NumPy Array
                raw_dataset = np.concatenate(
                    (np.array(ids)[:, np.newaxis], splitted_feats),
                    axis=1
                )
            else: # PyTorch Tensor
                raw_dataset = torch.cat(
                    (torch.unsqueeze(torch.tensor(ids), 1), splitted_feats),
                    dim=1
                )
            curr_dataset = cls(
                role=dataset.role,
                raw_dataset=raw_dataset,
                dataset_type=dataset.dataset_type
            )
            splitted_datasets.append(curr_dataset)

        return splitted_datasets

    @classmethod
    def buildin_dataset(cls, role, dataset_name, root, train,
                        passive_feat_frac, feat_perm_option,
                        download=False, transform=None, seed=None):
        def _check_params():
            assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), "Invalid role"
            assert dataset_name in Const.BUILDIN_DATASETS, "not supported dataset right now"
            assert 0 <= passive_feat_frac < 1, "passive_feat_frac should be in range [0, 1)"
            assert feat_perm_option in (Const.RANDOM,
                                        Const.SEQUENCE,
                                        Const.IMPORTANCE), \
                "invalid feat_perm_option, please check it again"

        # function body
        _check_params()
        np_dataset = cls._load_buildin_dataset(
            role=role, name=dataset_name,
            root=root, train=train, download=download,
            frac=passive_feat_frac, perm_option=feat_perm_option,
            seed=seed
        )
        if dataset_name in Const.DATA_TYPE_DICT[Const.REGRESSION]:
            dataset_type = Const.REGRESSION
        else:
            dataset_type = Const.CLASSIFICATION

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def dummy_dataset(cls, role, dataset_type,
                      n_samples, n_features, passive_feat_frac,
                      transform=None, seed=None
    ):
        if seed is not None:
            np.random.seed(seed)
        _ids = np.arange(n_samples)
        _feats = np.random.rand(n_samples, n_features)

        num_passive_feats = int(passive_feat_frac * n_features)
        if role == Const.PASSIVE_NAME:
            np_dataset = np.concatenate(
                (_ids[:, np.newaxis], _feats[:, :num_passive_feats]),
                axis=1
            )
        else:
            if dataset_type == Const.CLASSIFICATION:
                _labels = np.random.choice([0, 1], size=n_samples, replace=True)
            else:
                _labels = np.random.rand(n_samples)
            np_dataset = np.concatenate(
                (_ids[:, np.newaxis], _labels[:, np.newaxis], _feats[:, num_passive_feats:]),
                axis=1
            )

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_mysql(cls, role, dataset_type,
                   host, user, password, database, table,
                   *,
                   transform=None, port=3306
    ):
        """Load dataset from MySQL database."""
        import pymysql

        connection = pymysql.connect(host=host,
                                     user=user,
                                     port=port,
                                     password=password,
                                     database=database,
                                     cursorclass=pymysql.cursors.DictCursor)
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("select * from {}".format(table))
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)

        np_dataset = cls._pandas2numpy(df_dataset)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_oracle(cls, role, dataset_type,
                    host, user, password, database, table,
                    *,
                    transform=None, port=1521
    ):
        import cx_Oracle

        service_name = database
        dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
        connection = cx_Oracle.connect(user=user,
                                       password=password,
                                       dsn=dsn,
                                       encoding="UTF-8")
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("select * from {}".format(table))
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)

        np_dataset = cls._pandas2numpy(df_dataset)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_gaussdb(cls, role, dataset_type,
                     host, user, password, database, table,
                     *,
                     transform=None, port=6789
    ):
        """Load dataset from Gaussdb database."""
        # Note that Python is a dynamic programming language, so this error message can
        # be safely ignored in case where you do not need to call this method with a
        # Python interpreter without psyconpg2 package installed. If you do get
        # annoyed by this error message, you can use pip3 to install the psyconpg2 package
        # to suppress it. But the package installed via pip3 may be incompatible when
        # connnecting to gaussdb, this is why it is not included in LinkeFL's
        # requirements. If your application needs to load data from gaussdb, it's required
        # that you should first install guassdb manually and generate psycopg2 package
        # which can be imported by a Python script, and then use this method to load raw
        # data from guassdb into LinkeFL project.
        import psycopg2
        connection = psycopg2.connect(database=database,
                                      user=user,
                                      password=password,
                                      host=host,
                                      port=port)
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("select * from {}".format(table))
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)

        np_dataset = cls._pandas2numpy(df_dataset)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_gbase8a(cls, role, dataset_type,
                     host, user, password, database, table,
                     *,
                     transform=None, port=6789
    ):
        """Load dataset from gbase8a database."""
        import pymysql

        connection = pymysql.connect(database=database,
                                     user=user,
                                     password=password,
                                     host=host,
                                     port=port)
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("select * from {}".format(table))
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)

        np_dataset = cls._pandas2numpy(df_dataset)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_csv(cls, role, abs_path, dataset_type, transform=None):
        df_dataset = pd.read_csv(abs_path, delimiter=',', header=None)

        np_dataset = cls._pandas2numpy(df_dataset)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_excel(cls, role, abs_path, dataset_type, transform=None):
        """ Load dataset from excel.
        need dependency package openpyxl, support .xls .xlsx
        """
        df_dataset = pd.read_excel("{}".format(abs_path), index_col=False)

        np_dataset = cls._pandas2numpy(df_dataset)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @property
    def ids(self):  # read only
        """Always return a Python list"""
        # avoid re-computing on each function call
        if not hasattr(self, '_ids'):
            raw_ids = self._raw_dataset[:, 0]
            # the data type of _ids must be native Python integers, so it can be
            # compatible with the Private Set Intersection(PSI) module
            py_ids = [int(_id.item()) for _id in raw_ids]
            setattr(self, '_ids', py_ids)
        return getattr(self, '_ids')

    @property
    def features(self):  # read only
        if not hasattr(self, '_features'):
            if self.role == Const.ACTIVE_NAME:
                setattr(self, '_features', self._raw_dataset[:, 2:])
            else:
                setattr(self, '_features', self._raw_dataset[:, 1:])
        return getattr(self, '_features')

    @property
    def labels(self):  # read only
        if self.role == Const.PASSIVE_NAME:
            raise AttributeError('Passive party has no labels.')

        if not hasattr(self, '_labels'):
            raw_labels = self._raw_dataset[:, 1]
            if self.dataset_type == Const.REGRESSION:  # regression dataset
                setattr(self, '_labels', raw_labels)

            else:  # classification dataset, neet to convert label values to integers
                if isinstance(raw_labels, np.ndarray):
                    raw_labels = raw_labels.astype(np.int32) # NumPy
                else:
                    # the dtype of _labels should be cast to torch.long, otherwise,
                    # "the RuntimeError: expected scalar type Long but found Int"
                    # will be raised
                    raw_labels = raw_labels.type(torch.long) # PyTorch
                setattr(self, '_labels', raw_labels)
        return getattr(self, '_labels')

    @property
    def n_features(self):  # read only
        return self.features.shape[1]

    @property
    def n_samples(self):  # read only
        return self.features.shape[0]

    def filter(self, intersect_ids: Union[list, np.ndarray]):
        # Solution 1: this works only when dataset ids start from zero
        # if type(intersect_ids) == list:
        #     intersect_ids = np.array(intersect_ids)
        # self.np_dataset = self.np_dataset[intersect_ids]

        # Solution 2: this works only when dataset ids are sorted and ascending
        # reference: https://stackoverflow.com/a/12122989/8418540
        # if type(intersect_ids) == list:
        #     intersect_ids = np.array(intersect_ids)
        # all_ids = np.array(self.ids)
        # idxes = np.searchsorted(all_ids, intersect_ids)
        # self.np_dataset = self.np_dataset[idxes]

        # Solution 3: more robust, but slower
        if isinstance(intersect_ids, list):
            pass
        elif isinstance(intersect_ids, np.ndarray):
            intersect_ids = intersect_ids.tolist()
        else:
            raise TypeError('intersect_ids dtype is expected to be list or np.ndarray,'
                            'but got {}'.format(type(intersect_ids)))

        idxes = []
        all_ids = np.array(self.ids)
        for id_value in intersect_ids:
            idx = np.where(all_ids == id_value)[0][0]
            idxes.append(idx)
        new_raw_dataset = self._raw_dataset[idxes]
        self.set_dataset(new_raw_dataset)

    def describe(self):
        import seaborn as sns

        from matplotlib import pyplot as plt
        from termcolor import colored

        print(colored('Number of samples: {}'.format(self.n_samples), 'red'))
        print(colored('Number of features: {}'.format(self.n_features), 'red'))
        if self.role == Const.ACTIVE_NAME and len(np.unique(list(self.labels))) == 2:
            if isinstance(self.labels, np.ndarray): # Numpy Array
                n_positive = (self.labels == 1).astype(int).sum()
            else: # PyTorch Tensor
                n_positive = (self.labels == 1).type(torch.int32).sum().item()
            n_negative = self.n_samples - n_positive
            print(colored('Positive samples: Negative samples = {}:{}'
                          .format(n_positive, n_negative), 'red'))
        print()

        # Output of statistical values of the data set.
        pd.set_option('display.max_columns', None)
        df_dataset = pd.DataFrame(self._raw_dataset)
        if self.role == Const.ACTIVE_NAME:
            df_dataset.rename(columns={0: 'id', 1: 'lable'}, inplace=True)
            for i in range(self.n_features):
                df_dataset.rename(columns={i + 2: 'x' + str(i + 1)}, inplace=True)
        elif self.role == Const.PASSIVE_NAME:
            df_dataset.rename(columns={0: 'id'}, inplace=True)
            for i in range(self.n_features):
                df_dataset.rename(columns={i + 1: 'x' + str(i + 1)}, inplace=True)

        print(colored('The first 5 rows and the last 5 rows of the dataset are as follows:', 'red'))
        print(pd.concat([df_dataset.head(), df_dataset.tail()]))
        print()

        print(colored('The information about the dataset including the index '
                      'dtype and columns, non-null values and memory usage '
                      'are as follows:', 'red'))
        df_dataset.info()
        print()

        print(colored('The descriptive statistics include those that summarize '
                      'the central tendency, dispersion and shape of the dataset’s '
                      'distribution, excluding NaN values are as follows:', 'red'))
        col_names = list(df_dataset.columns.values)
        num_unique_data = np.array(df_dataset[col_names].nunique().values)
        num_unique = pd.DataFrame(data=num_unique_data.reshape((1, -1)),
                                  index=['unique'],
                                  columns=col_names)
        print(pd.concat([df_dataset.describe(), num_unique]))
        print()

        # Output the distribution for the data label.
        if self.role == Const.ACTIVE_NAME:
            n_classes = len(np.unique(list(self.labels)))
            if self.dataset_type == Const.REGRESSION:  # regression dataset
                dis_label = pd.DataFrame(data=self.labels.reshape((-1, 1)),
                                         columns=['label'])
                # histplot
                sns.histplot(dis_label, kde=True, linewidth=1)
            else:  # classification dataset
                bars = [str(i) for i in range(n_classes)]
                if isinstance(self.labels, np.ndarray): # Numpy Array
                    counts = [(self.labels == i).astype(int).sum()
                                for i in range(n_classes)]
                else: # PyTorch Tensor
                    counts = [(self.labels == i).type(torch.int32).sum().item()
                                for i in range(n_classes)]
                x = np.arange(len(bars))
                width = 0.5 / n_classes

                # barplot
                rec = plt.bar(x, counts, width=width)
                # show corresponding value of the bar on top of itself
                for bar in rec:
                    h = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, h, h,
                             ha='center',
                             va='bottom',
                             size=14)
                plt.xticks(x, bars, fontsize=14)

            plt.show()

    def get_dataset(self):
        return self._raw_dataset

    def set_dataset(self, new_raw_dataset: Union[np.ndarray, torch.Tensor]):
        # must delete old properties to save memory
        if hasattr(self, '_raw_dataset'):
            del self._raw_dataset
        if hasattr(self, '_ids'):
            del self._ids
        if hasattr(self, '_features'):
            del self._features
        if hasattr(self, '_labels'):
            del self._labels

        # update new property
        self._raw_dataset = new_raw_dataset

    @staticmethod
    def _load_buildin_dataset(role, name,
                              root, train, download,
                              frac, perm_option,
                              seed=None
    ):
        import os
        from urllib.error import URLError

        from linkefl.feature import cal_importance_ranking
        from linkefl.util import urlretrive

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
            "avazu": ("avazu-train.csv", "avazu-test.csv")
        }
        BASE_URL = 'http://47.96.163.59:80/datasets/'
        root = os.path.join(root, 'tabular')

        if download:
            if _check_exists(name, root, train, resources):
                # if data files have already been downloaded, then skip this branch
                print('Data files have already been downloaded.')
            else:
                # download data files from web server
                os.makedirs(root, exist_ok=True)
                filename = resources[name][0] if train else resources[name][1]
                fpath = os.path.join(root, filename)
                full_url = BASE_URL + filename
                try:
                    print('Downloading {} to {}'.format(full_url, fpath))
                    urlretrive(full_url, fpath)
                except URLError as error:
                    raise RuntimeError('Failed to download {} with error message: {}'
                                       .format(full_url, error))
                print('Done!')
        if not _check_exists(name, root, train, resources):
            raise RuntimeError('Dataset not found. You can use download=True to get it.')

        # ===== 1. Load dataset =====
        if train:
            np_csv = np.genfromtxt(
                os.path.join(root, resources[name][0]),
                delimiter=',',
                encoding="utf-8"
            )
        else:
            np_csv = np.genfromtxt(
                os.path.join(root, resources[name][1]),
                delimiter=',',
                encoding="utf-8"
            )
        _ids = np_csv[:, 0] # no need to convert to integers here
        _labels = np_csv[:, 1] # no need to convert to integers here
        _feats = np_csv[:, 2:]

        # ===== 2. Apply feature permutation =====
        if perm_option == Const.SEQUENCE:
            permuted_feats = _feats
        elif perm_option == Const.RANDOM:
            if seed is not None:
                np.random.seed(seed)
            permuted_feats = _feats[:, np.random.permutation(_feats.shape[1])]
            del _feats  # save memory
        elif perm_option == Const.IMPORTANCE:
            rankings = cal_importance_ranking(name, _feats, _labels)
            permuted_feats = _feats[:, rankings]
        else:
            raise ValueError('Invalid permutation option.')

        # ===== 3. Split feature =====
        num_passive_feats = int(frac * permuted_feats.shape[1])
        if role == Const.PASSIVE_NAME:
            splitted_feats = permuted_feats[:, :num_passive_feats]
            np_dataset = np.concatenate(
                (_ids[:, np.newaxis], splitted_feats),
                axis=1
            )
        else:
            splitted_feats = permuted_feats[:, num_passive_feats:]
            np_dataset = np.concatenate(
                (_ids[:, np.newaxis], _labels[:, np.newaxis], splitted_feats),
                axis=1
            )

        return np_dataset

    @staticmethod
    def _pandas2numpy(df_dataset: pd.DataFrame):
        """Transform a pandas DataFrame into Numpy Array"""
        from pandas.core.dtypes.common import is_numeric_dtype

        for i, dtype in enumerate(df_dataset.dtypes):
            if not is_numeric_dtype(dtype):
                df_dataset[df_dataset.columns[i]], _ = pd.factorize(df_dataset.iloc[:, i])

        np_dataset = df_dataset.to_numpy()
        return np_dataset
