from __future__ import annotations  # python >= 3.7, give type hint before definition

import io
import random
from typing import Union

import numpy as np
import pandas as pd
import torch

from linkefl.base import BaseTransformComponent
from linkefl.common.const import Const


class CommonDataset:
    """Common Dataset"""
    mappings = None  # mappings for pandas non-numeric columns

    def __init__(self,
                 role: str,
                 raw_dataset: Union[np.ndarray, torch.Tensor],
                 dataset_type: str,
                 transform: BaseTransformComponent = None,
    ):
        super(CommonDataset, self).__init__()
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
                        download=False, transform=None, seed=None
    ):
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
            random.seed(seed)
        _ids = np.arange(n_samples)
        _feats = [random.random() for _ in range(n_samples * n_features)]
        _feats = np.array(_feats).reshape(n_samples, n_features)

        num_passive_feats = int(passive_feat_frac * n_features)
        if role == Const.PASSIVE_NAME:
            np_dataset = np.concatenate(
                (_ids[:, np.newaxis], _feats[:, :num_passive_feats]),
                axis=1
            )
        else:
            if dataset_type == Const.CLASSIFICATION:
                _labels = np.array([random.choice([0, 1]) for _ in range(n_samples)])
            else:
                _labels = np.array([random.random() for _ in range(n_samples)])
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
                   target_fields=None, excluding_fields=False,
                   mappings=None, transform=None, port=3306
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
                selected_fields = cls._get_selected_fields(
                    db_type='mysql',
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields
                )
                sql = "select" + " " + ",".join(selected_fields) + " " + "from {}".format(table)
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)

        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

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
                    target_fields=None, excluding_fields=False,
                    mappings=None, transform=None, port=1521
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
                selected_fields = cls._get_selected_fields(
                    db_type='oracle',
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields
                )
                sql = "select" + " " + ",".join(selected_fields) + " " + "from {}".format(table)
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)

        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

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
                     target_fields=None, excluding_fields=False,
                     mappings=None, transform=None, port=6789
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
                selected_fields = cls._get_selected_fields(
                    db_type='gaussdb',
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields
                )
                sql = "select" + " " + ",".join(selected_fields) + " " + "from {}".format(table)
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)

        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

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
                     target_fields=None, excluding_fields=False,
                     mappings=None, transform=None, port=6789
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
                selected_fields = cls._get_selected_fields(
                    db_type='gbase8a',
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields
                )
                sql = "select" + " " + ",".join(selected_fields) + " " + "from {}".format(table)
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)

        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_csv(cls, role, abs_path, dataset_type, delimiter=',', mappings=None, transform=None):
        df_dataset = pd.read_csv(
            abs_path,
            delimiter=delimiter,
            header=None,
            skipinitialspace=True, # skip spaces after delimiter
        )

        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_excel(cls, role, abs_path, dataset_type, mappings=None, transform=None):
        """ Load dataset from excel.
        need dependency package openpyxl, support .xls .xlsx
        """
        df_dataset = pd.read_excel("{}".format(abs_path), index_col=False)

        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_url(cls, role, url, dataset_type, delimiter=',', mappings=None, transform=None):
        import requests

        # do not directly use pd.read_csv(url),
        # because it will fail if it requires authentication
        data_raw = requests.get(url).content
        data_byte = io.StringIO(data_raw.decode('utf-8'))
        df_dataset = pd.read_csv(
            data_byte,
            delimiter=delimiter,
            header=None,
            skipinitialspace=True, # skip spaces after delimiter
        )

        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            dataset_type=dataset_type,
            transform=transform
        )

    @classmethod
    def from_anyfile(cls, role, abs_path, dataset_type, mappings=None, transform=None):
        extension = abs_path.split('.')[-1]
        if extension in ('csv', 'txt', 'dat'):
            # read first two lines to determine the delimiter
            with open(abs_path) as f:
                first_line = f.readline()
                second_line = f.readline()
            if "," in first_line or "," in second_line:
                delimiter = ","
            else:
                delimiter = "\s+" # regular expression, indicating one or more whitespace
            return cls.from_csv(
                role=role,
                abs_path=abs_path,
                dataset_type=dataset_type,
                delimiter=delimiter,
                mappings=mappings,
                transform=transform
            )

        elif extension in ('xls', 'xlsx'):
            return cls.from_excel(
                role=role,
                abs_path=abs_path,
                dataset_type=dataset_type,
                mappings=mappings,
                transform=transform
            )

        else:
            raise RuntimeError("file type {} is not supported.".format(extension))

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

    def obfuscated_ids(self, option='md5'):
        import hashlib

        assert option in ('md5', 'sha256'), \
            "ids obfuscation option can only take from md5 or sha256"
        if option == 'md5':
            obfuscate_func = hashlib.md5
        else:
            obfuscate_func = hashlib.sha256

        raw_ids = self.ids
        obfuscated_ids = []
        for _id in raw_ids:
            _id_encode = str(_id).encode()
            _id_hash = obfuscate_func(_id_encode).hexdigest()
            _id_int = int(_id_hash, 16)
            obfuscated_ids.append(_id_int)
        return obfuscated_ids

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
                random.seed(seed)
            perm = list(range(_feats.shape[1]))
            random.shuffle(perm)
            permuted_feats = _feats[:, perm]
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
    def _pandas2numpy(df_dataset: pd.DataFrame, mappings: dict = None):
        """Transform a pandas DataFrame into Numpy Array"""
        from pandas.core.dtypes.common import is_numeric_dtype

        if mappings is None:
            create_mappings_flag = True
            mappings = dict()
        else:
            create_mappings_flag = False
            mappings = mappings

        for i, dtype in enumerate(df_dataset.dtypes):
            if not is_numeric_dtype(dtype):
                if create_mappings_flag is True:
                    df_dataset[df_dataset.columns[i]], uniques = pd.factorize(df_dataset.iloc[:, i])
                    mapping = dict(zip(uniques, range(len(uniques))))
                    mappings[i] = mapping
                else:
                    df_dataset[df_dataset.columns[i]] = df_dataset.iloc[:, i].replace(mappings[i])

        CommonDataset.mappings = mappings

        np_dataset = df_dataset.to_numpy()
        return np_dataset

    @staticmethod
    def _get_selected_fields(db_type, cursor, table, target_fields, excluding_fields):
        if db_type == "oracle":
            sql = "select * from {} fetch first 1 rows only".format(table)
        else:
            sql = "select * from {} limit 1".format(table)
        cursor.execute(sql)
        # description is a tuple of tuple,
        # the first position of tuple element is the field name
        all_fields = [tuple_[0] for tuple_ in cursor.description]

        if target_fields is None:
            selected_fields = all_fields
        else:
            if not excluding_fields:
                selected_fields = target_fields
            else:
                selected_fields = list(set(all_fields) - set(target_fields))

        return selected_fields


if __name__ == "__main__":
    from linkefl.feature.transform import OneHot

    print("the first df_dataset")
    _df_dataset = pd.DataFrame(
        {"id": [0, 1, 2], "x": [1.1, 1.2, 1.3], "a": ["aaa", "bbb", "ccc"], "b": ["aa", "bb", "cc"]}
    )
    print(_df_dataset)
    _np_dataset = CommonDataset._pandas2numpy(_df_dataset)
    _mappings = CommonDataset.mappings
    # you can save these mappings and load it back when loading testset at inference pahse
    # with open('train_mappings.pkl', 'wb') as f:
    #     pickle.dump(mappings, f)
    print(_np_dataset)
    _np_dataset = OneHot([1, 2]).fit(_np_dataset, Const.PASSIVE_NAME)
    print(_np_dataset)

    print()

    print("the second df_dataset")
    another_df_dataset = pd.DataFrame(
        {"id": [0, 1, 2], "x": [1.1, 1.2, 1.3], "a": ["bbb", "ccc", "aaa"], "b": ["cc", "aa", "bb"]}
    )
    print(another_df_dataset)
    # you can load the mappings back and apply it to testset
    # with open('train_mappings.pkl', 'rb') as f:
    #     mappings = pickle.load(f)
    another_np_dataset = CommonDataset._pandas2numpy(another_df_dataset, mappings=_mappings)
    print(another_np_dataset)
    another_np_dataset = OneHot([1, 2]).fit(another_np_dataset, Const.PASSIVE_NAME)
    print(another_np_dataset)
