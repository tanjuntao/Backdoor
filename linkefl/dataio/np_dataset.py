import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_diabetes,
    load_iris,
    load_wine,
)
import seaborn as sns
from sklearn.model_selection import train_test_split
from termcolor import colored

from linkefl.common.const import Const
from linkefl.dataio.base import BaseDataset
from linkefl.feature import cal_importance_ranking


class NumpyDataset(BaseDataset):
    def __init__(self, role, abs_path=None, transform=None, existing_dataset=None):
        super(NumpyDataset, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.role = role

        # if existing_dataset is None:
        #     if abs_path is not None:
        #         # self._np_dataset = np.genfromtxt(abs_path, delimiter=',', encoding="utf-8")
        #         self._np_dataset = pd.read_csv(abs_path, delimiter=',', header=None)
        #     else:
        #         raise Exception('abs_path should not be None')
        # else:
        #     self.set_dataset(existing_dataset)

        np_data = NumpyDataset._load_csv_dataset(abs_path, existing_dataset)
        # after this function call, self._np_dataset property will be created
        self.set_dataset(np_data)

        if transform is not None:
            self._np_dataset = transform(self._np_dataset, role=role)
        self.has_label = True if role == Const.ACTIVE_NAME else False

    @classmethod
    def train_test_split(cls, whole_dataset, test_size, seed=1314):
        """Split the whole np_dataset into trainset and testset according to specific seed"""
        assert isinstance(whole_dataset, NumpyDataset), 'whole_dataset should be' \
                                                        'an instance of NumpyDataset'
        assert 0 <= test_size < 1, 'validate size should be in range (0, 1)'

        n_train_samples = int(whole_dataset.n_samples * (1 - test_size))
        np.random.seed(seed)
        perm = np.random.permutation(whole_dataset.n_samples)
        np_trainset = whole_dataset.get_dataset()[perm[:n_train_samples], :]
        np_testset = whole_dataset.get_dataset()[perm[n_train_samples:], :]

        trainset = cls(role=whole_dataset.role, existing_dataset=np_trainset)
        testset = cls(role=whole_dataset.role, existing_dataset=np_testset)

        return trainset, testset


    @classmethod
    def feature_split(cls, whole_dataset, n_splits, option=Const.SEQUENCE, seed=1314):
        if option == Const.SEQUENCE:
            perm = np.arange(whole_dataset.n_features)
        elif option == Const.RANDOM:
            np.random.seed(seed)
            perm = np.random.permutation(whole_dataset.n_features)
        else:
            raise ValueError('in feature_split method, the option of feature '
                             'permutation can only take from SEQUENCE AND RANDOM.')

        permed_features = whole_dataset.features[:, perm]
        ids = np.array(whole_dataset.ids)
        step = whole_dataset.n_features // n_splits

        splitted_datasets = []
        for i in range(n_splits):
            begin_idx = i * step
            if i != n_splits - 1:
                end_idx = (i + 1) * step
            else:
                end_idx = whole_dataset.n_features
            splitted_feats = permed_features[:, begin_idx:end_idx]
            np_dataset = np.concatenate((ids[:, np.newaxis], splitted_feats), axis=1)
            curr_dataset = cls(role=whole_dataset.role, existing_dataset=np_dataset)
            splitted_datasets.append(curr_dataset)

        return splitted_datasets


    @classmethod
    def buildin_dataset(cls,
                        role,
                        dataset_name,
                        train,
                        passive_feat_frac,
                        feat_perm_option,
                        transform=None,
                        seed=1314):
        def _check_params():
            assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
            assert dataset_name in Const.BUILDIN_DATASETS, "not supported dataset right now"
            assert 0 <= passive_feat_frac < 1, "passive_feat_frac should be in range (0, 1)"
            assert feat_perm_option in (Const.RANDOM,
                                        Const.SEQUENCE,
                                        Const.IMPORTANCE), \
                "invalid feat_perm_option, please check it again"

        # function body
        _check_params()
        np_dataset = NumpyDataset._load_buildin_dataset(role=role,
                                                        name=dataset_name,
                                                        train=train,
                                                        frac=passive_feat_frac,
                                                        perm_option=feat_perm_option,
                                                        seed=seed)

        return cls(role=role, transform=transform, existing_dataset=np_dataset)

    @classmethod
    def database_dataset(cls,
                         role,
                         host,
                         user,
                         password,
                         database,
                         table,
                         port=3306):
        """Load dataset from MySQL database."""
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

        return cls(role=role, existing_dataset=df_dataset)

        #         count = 0
        #         if ids is None:
        #             sql = "SELECT * FROM `{}` ".format(table)
        #             cursor.execute(sql, ())
        #             result = cursor.fetchall()
        #             final_dataframe = pd.DataFrame.from_dict(result)
        #
        #         else:
        #             for id in ids:
        #                 sql = "SELECT * FROM `{}` WHERE `id`=%s".format(table)
        #                 cursor.execute(sql, (id,))
        #                 item = cursor.fetchone()
        #
        #                 if item == None:
        #                     raise ValueError("Wrong input id:{}".format(id))
        #
        #                 if count == 0:
        #                     final_dataframe = pd.DataFrame.from_dict([item])
        #                     count += 1
        #                 else:
        #                     item_dataframe = pd.DataFrame.from_dict([item])
        #                     final_dataframe = final_dataframe.append(item_dataframe)
        #
        # return cls(role=role, existing_dataset=final_dataframe)


    @staticmethod
    def _load_csv_dataset(path, existing_dataset=None):
        if existing_dataset is None:
            if path is not None:
                # TODO: support one-hot encoding here
                np_dataset = np.genfromtxt(path, delimiter=',', encoding="utf-8")
                # np_dataset = pd.read_csv(path, delimiter=',', header=None)
            else:
                raise Exception('CSV file path is not provided')
        else:
            np_dataset = existing_dataset # just assign a new pointer

        return np_dataset

    # utility method
    @staticmethod
    def _load_buildin_dataset(role, name, train, frac, perm_option, seed):
        curr_path = os.path.abspath(os.path.dirname(__file__))

        # 1. load whole dataset and split it into trainset and testset
        if name == 'cancer':  # classification
            cancer = load_breast_cancer()
            x_train, x_test, y_train, y_test = train_test_split(cancer.data,
                                                                cancer.target,
                                                                test_size=0.2,
                                                                random_state=0)

            if train:
                _ids = np.arange(x_train.shape[0])
                _feats = x_train
                _labels = y_train
            else:
                _ids = np.arange(x_train.shape[0],
                                 x_train.shape[0] + x_test.shape[0])
                _feats = x_test
                _labels = y_test

            # cancer = load_breast_cancer()
            # _whole_feats = cancer.data
            # _whole_labels = cancer.target
            # _n_samples = len(_whole_labels)
            # _whole_ids = np.arange(_n_samples)
            # np.random.seed(seed)
            # shuffle = np.random.permutation(_n_samples)
            # test_size = 0.2
            # _n_train_samples = int(_n_samples * (1 - test_size))
            # if train:
            #     _ids = _whole_ids[shuffle[:_n_train_samples]]
            #     _feats = _whole_feats[shuffle[:_n_train_samples], :]
            #     _labels = _whole_labels[shuffle[:_n_train_samples]]
            # else:
            #     _ids = _whole_ids[shuffle[_n_train_samples:]]
            #     _feats = _whole_feats[shuffle[_n_train_samples:], :]
            #     _labels = _whole_labels[shuffle[_n_train_samples:]]

        elif name == 'digits':  # classification
            X, Y = load_digits(return_X_y=True)
            odd_idxes = np.where(Y % 2 == 1)[0]
            even_idxes = np.where(Y % 2 == 0)[0]
            Y[odd_idxes] = 1
            Y[even_idxes] = 0
            x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                                test_size=0.2,
                                                                random_state=0)
            if train:
                _ids = np.arange(x_train.shape[0])
                _feats = x_train
                _labels = y_train
            else:
                _ids = np.arange(x_train.shape[0],
                                 x_train.shape[0] + x_test.shape[0])
                _feats = x_test
                _labels = y_test

            # _whole_feats, _whole_labels = load_digits(return_X_y=True)
            # _n_samples = len(_whole_labels)
            # odd_idxes = np.where(_whole_labels % 2 == 1)[0]
            # even_idxes = np.where(_whole_labels % 2 == 0)[0]
            # _whole_labels[odd_idxes] = 1
            # _whole_labels[even_idxes] = 0
            # _whole_ids = np.arange(len(_whole_labels))
            # np.random.seed(seed)
            # shuffle = np.random.permutation(_n_samples)
            # test_size = 0.2
            # _n_train_samples = int(_n_samples * (1 - test_size))
            # if train:
            #     _ids = _whole_ids[shuffle[:_n_train_samples]]
            #     _feats = _whole_feats[shuffle[:_n_train_samples], :]
            #     _labels = _whole_labels[shuffle[:_n_train_samples]]
            # else:
            #     _ids = _whole_ids[shuffle[_n_train_samples:]]
            #     _feats = _whole_feats[shuffle[_n_train_samples:], :]
            #     _labels = _whole_labels[shuffle[_n_train_samples:]]

        elif name == 'diabetes':  # regression
            # original dataset shape: 442*10
            _whole_feats, _whole_labels = load_diabetes(return_X_y=True,
                                                        scaled=True)
            _n_samples = len(_whole_labels)
            _whole_ids = np.arange(_n_samples)
            test_size = 40  # fixed testing set size
            if train:
                _ids = _whole_ids[:-test_size]
                _feats = _whole_feats[:-test_size]
                _labels = _whole_labels[:-test_size]
            else:
                _ids = _whole_ids[-test_size:]
                _feats = _whole_feats[-test_size:]
                _labels = _whole_labels[-test_size:]

        elif name == 'iris': # classification, 3 classes
            X, Y = load_iris(return_X_y=True)
            x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                                test_size=0.2,
                                                                random_state=0)
            if train:
                _ids = np.arange(x_train.shape[0])
                _feats = x_train
                _labels = y_train
            else:
                _ids = np.arange(x_train.shape[0], x_train.shape[0] + x_test.shape[0])
                _feats = x_test
                _labels = y_test

        elif name == 'wine': # classification, 3 classes
            X, Y = load_wine(return_X_y=True)
            x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                                test_size=0.2,
                                                                random_state=0)
            if train:
                _ids = np.arange(x_train.shape[0])
                _feats = x_train
                _labels = y_train
            else:
                _ids = np.arange(x_train.shape[0],
                                 x_train.shape[0] + x_test.shape[0])
                _feats = x_test
                _labels = y_test

        elif name == 'epsilon':  # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/epsilon_train.csv'),
                    delimiter=',', encoding="utf-8")
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/epsilon_test.csv'),
                    delimiter=',', encoding="utf-8")
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'census':  # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/census_income_train.csv'),
                    delimiter=',', encoding="utf-8")
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/census_income_test.csv'),
                    delimiter=',', encoding="utf-8")
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'credit':  # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/give_me_some_credit_train.csv'),
                    delimiter=',', encoding="utf-8")
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/give_me_some_credit_test.csv'),
                    delimiter=',', encoding="utf-8")
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'default_credit':  # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/default_credit_train.csv'),
                    delimiter=',', encoding="utf-8")
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/default_credit_test.csv'),
                    delimiter=',', encoding="utf-8")
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'covertype':  # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/covertype_train.csv'),
                    delimiter=',', encoding="utf-8")
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/covertype_test.csv'), 
                    delimiter=',', encoding="utf-8")
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'higgs':  # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/higgs_train.csv'),
                    delimiter=',', encoding="utf-8")
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/higgs_test.csv'), 
                    delimiter=',', encoding="utf-8")
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'year':  # regression
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/year_train.csv'),
                    delimiter=',', encoding="utf-8")
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/year_test.csv'), 
                    delimiter=',', encoding="utf-8")
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'nyc-taxi':  # regression
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/nyc-taxi_train.csv'),
                    delimiter=',', encoding="utf-8")
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path,
                                 '../data/tabular/nyc-taxi_test.csv'), 
                    delimiter=',', encoding="utf-8")
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        else:
            raise ValueError('Invalid dataset name.')

        # 2. Apply feature permutation to the train features or validate features
        if perm_option == Const.SEQUENCE:
            permuted_feats = _feats
        elif perm_option == Const.RANDOM:
            np.random.seed(seed)
            permuted_feats = _feats[:, np.random.permutation(_feats.shape[1])]
            del _feats  # save memory
        elif perm_option == Const.IMPORTANCE:
            rankings = cal_importance_ranking(name, _feats, _labels)
            permuted_feats = _feats[:, rankings]
        else:
            raise ValueError('Invalid permutation option.')

        # 3. Split the features into active party and passive party
        num_passive_feats = int(frac * permuted_feats.shape[1])
        if role == Const.PASSIVE_NAME:
            splitted_feats = permuted_feats[:, :num_passive_feats]
            np_dataset = np.concatenate((_ids[:, np.newaxis], splitted_feats),
                                        axis=1)
        else:
            splitted_feats = permuted_feats[:, num_passive_feats:]
            np_dataset = np.concatenate(
                (_ids[:, np.newaxis], _labels[:, np.newaxis], splitted_feats),
                axis=1)

        return np_dataset

    @property
    def ids(self): # read only
        # avoid re-computing on each function call
        if not hasattr(self, '_ids'):
            np_ids = self._np_dataset[:, 0].astype(np.int32)
            # data type of ids must be python build-in integer, not numpy integer
            py_ids = [_id.item() for _id in np_ids]
            setattr(self, '_ids', py_ids)
        return getattr(self, '_ids')

    @property
    def features(self): # read only
        if not hasattr(self, '_features'):
            if self.role == Const.ACTIVE_NAME:
                setattr(self, '_features', self._np_dataset[:, 2:])
            else:
                setattr(self, '_features', self._np_dataset[:, 1:])
        return getattr(self, '_features')

    @property
    def labels(self): # read only
        if self.role == Const.PASSIVE_NAME:
            raise AttributeError('Passive party has no labels.')

        if not hasattr(self, '_labels'):
            labels = self._np_dataset[:, 1]
            # simply treat dataset where the label column contains more than
            # 100 unique values as regression dataset
            if len(np.unique(labels)) > 100: # regression dataset
                setattr(self, '_labels', labels)
            else: # classification dataset
                # convert potential floating-point label values, e.g., 4.0, to
                # int data type
                setattr(self, '_labels', labels.astype(np.int32))
        return getattr(self, '_labels')

    @property
    def n_features(self): # read only
        return self.features.shape[1]

    @property
    def n_samples(self): # read only
        return self.features.shape[0]

    def describe(self):
        print(colored('Number of samples: {}'.format(self.n_samples), 'red'))
        print(colored('Number of features: {}'.format(self.n_features), 'red'))
        if self.role == Const.ACTIVE_NAME and len(np.unique(self.labels)) == 2:
            n_positive = (self.labels == 1).astype(np.int32).sum()
            n_negative = self.n_samples - n_positive
            print(colored('Positive samples: Negative samples = {}:{}'
                          .format(n_positive, n_negative), 'red'))
        print()

        # Output of statistical values of the data set.
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(self._np_dataset)
        if self.role == Const.ACTIVE_NAME:
            df.rename(columns={0:'id', 1:'lable'}, inplace=True)
            for i in range(self.n_features):
                df.rename(columns={i+2: 'x' + str(i+1)}, inplace=True)
        elif self.role == Const.PASSIVE_NAME:
            df.rename(columns={0: 'id'}, inplace=True)
            for i in range(self.n_features):
                df.rename(columns={i+1: 'x' + str(i+1)}, inplace=True)

        print(colored('The first 5 rows and the last 5 rows of the dataset are as follows:', 'red'))
        print(pd.concat([df.head(), df.tail()]))
        print()

        print(colored('The information about the dataset including the index '
                      'dtype and columns, non-null values and memory usage '
                      'are as follows:', 'red'))
        df.info()
        print()

        print(colored('The descriptive statistics include those that summarize '
                      'the central tendency, dispersion and shape of the dataset’s '
                      'distribution, excluding NaN values are as follows:', 'red'))
        col_names = df.columns.values.tolist()
        num_unique_data = np.array(df[col_names].nunique().values)
        num_unique = pd.DataFrame(data=num_unique_data.reshape((1, -1)),
                                  index=['unique'],
                                  columns=col_names)
        print(pd.concat([df.describe(), num_unique]))
        print()

        # Output the distribution for the data label.
        if self.role == Const.ACTIVE_NAME:
            n_classes = len(np.unique(self.labels))
            if n_classes > 100: # regression dataset
                dis_label = pd.DataFrame(data=self.labels.reshape((-1, 1)),
                                         columns=['label'])
                # histplot
                sns.histplot(dis_label, kde=True, linewidth=1)
            else: # classification dataset
                bars = [str(i) for i in range(n_classes)]
                counts = [(self.labels == i).astype(np.int32).sum() for i in range(n_classes)]
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


    def filter(self, intersect_ids):
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
        if type(intersect_ids) == np.ndarray:
            intersect_ids = intersect_ids.tolist()
        if type(intersect_ids) == list:
            pass

        idxes = []
        all_ids = np.array(self.ids)
        for _id in intersect_ids:
            idx = np.where(all_ids == _id)[0][0]
            idxes.append(idx)
        new_np_dataset = self._np_dataset[idxes] # return a new numpy object
        self.set_dataset(new_np_dataset)

    def get_dataset(self):
        return self._np_dataset

    def set_dataset(self, new_np_dataset):
        assert isinstance(new_np_dataset, np.ndarray),\
            "new_np_dataset should be an instance of np.ndarray"

        # must delete old properties to save memory
        if hasattr(self, '_np_dataset'):
            del self._np_dataset
        if hasattr(self, '_ids'):
            del self._ids
        if hasattr(self, '_features'):
            del self._features
        if hasattr(self, '_labels'):
            del self._labels

        # update new property
        self._np_dataset = new_np_dataset

'''
class BuildinNumpyDataset(NumpyDataset):
    def __init__(self,
                 dataset_name,
                 train,
                 role,
                 passive_feat_frac,
                 feat_perm_option,
                 transform=None,
                 seed=1314
    ):
        assert dataset_name in Const.BUILDIN_DATASETS, f"{dataset_name} is not a" \
                                                      f"build-in dataset"
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        assert 0 < passive_feat_frac < 1, "The feature fraction of passive party" \
                                          "should be in range (0, 1)"
        assert feat_perm_option in (Const.RANDOM, Const.SEQUENCE, Const.IMPORTANCE),\
            "The feature permutation option should be among random, sequence and importance"

        self.dataset_name = dataset_name
        self.role = role
        self.train = train
        self.passive_feat_frac = passive_feat_frac
        self.feat_perm_option = feat_perm_option
        self.seed = seed

        self._np_dataset = self._load_dataset(dataset_name, train, role,
                                             passive_feat_frac, feat_perm_option, seed)
        if transform is not None:
            self._np_dataset = transform(self._np_dataset)
        self.has_label = True if role == Const.ACTIVE_NAME else False

    def _load_dataset(self, name, train, role, frac, perm_option, seed):
        curr_path = os.path.abspath(os.path.dirname(__file__))

        # 1. load whole dataset and split it into trainset and testset
        if name == 'cancer': # classification
            cancer = load_breast_cancer()
            x_train, x_test, y_train, y_test = train_test_split(cancer.data,
                                                                cancer.target,
                                                                test_size=0.2,
                                                                random_state=0)

            if train:
                _ids = np.arange(x_train.shape[0])
                _feats = x_train
                _labels = y_train
            else:
                _ids = np.arange(x_train.shape[0], x_train.shape[0] + x_test.shape[0])
                _feats = x_test
                _labels = y_test

            # cancer = load_breast_cancer()
            # _whole_feats = cancer.data
            # _whole_labels = cancer.target
            # _n_samples = len(_whole_labels)
            # _whole_ids = np.arange(_n_samples)
            # np.random.seed(seed)
            # shuffle = np.random.permutation(_n_samples)
            # test_size = 0.2
            # _n_train_samples = int(_n_samples * (1 - test_size))
            # if train:
            #     _ids = _whole_ids[shuffle[:_n_train_samples]]
            #     _feats = _whole_feats[shuffle[:_n_train_samples], :]
            #     _labels = _whole_labels[shuffle[:_n_train_samples]]
            # else:
            #     _ids = _whole_ids[shuffle[_n_train_samples:]]
            #     _feats = _whole_feats[shuffle[_n_train_samples:], :]
            #     _labels = _whole_labels[shuffle[_n_train_samples:]]

        elif name == 'digits': # classification
            X, Y = load_digits(return_X_y=True)
            odd_idxes = np.where(Y % 2 == 1)[0]
            even_idxes = np.where(Y % 2 == 0)[0]
            Y[odd_idxes] = 1
            Y[even_idxes] = 0
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
            if train:
                _ids = np.arange(x_train.shape[0])
                _feats = x_train
                _labels = y_train
            else:
                _ids = np.arange(x_train.shape[0], x_train.shape[0] + x_test.shape[0])
                _feats = x_test
                _labels = y_test

            # _whole_feats, _whole_labels = load_digits(return_X_y=True)
            # _n_samples = len(_whole_labels)
            # odd_idxes = np.where(_whole_labels % 2 == 1)[0]
            # even_idxes = np.where(_whole_labels % 2 == 0)[0]
            # _whole_labels[odd_idxes] = 1
            # _whole_labels[even_idxes] = 0
            # _whole_ids = np.arange(len(_whole_labels))
            # np.random.seed(seed)
            # shuffle = np.random.permutation(_n_samples)
            # test_size = 0.2
            # _n_train_samples = int(_n_samples * (1 - test_size))
            # if train:
            #     _ids = _whole_ids[shuffle[:_n_train_samples]]
            #     _feats = _whole_feats[shuffle[:_n_train_samples], :]
            #     _labels = _whole_labels[shuffle[:_n_train_samples]]
            # else:
            #     _ids = _whole_ids[shuffle[_n_train_samples:]]
            #     _feats = _whole_feats[shuffle[_n_train_samples:], :]
            #     _labels = _whole_labels[shuffle[_n_train_samples:]]

        elif name == 'diabetes': # regression
            # original dataset shape: 442*10
            _whole_feats, _whole_labels = load_diabetes(return_X_y=True, scaled=True)
            _n_samples = len(_whole_labels)
            _whole_ids = np.arange(_n_samples)
            test_size = 40 # fixed testing set size
            if train:
                _ids = _whole_ids[:-test_size]
                _feats = _whole_feats[:-test_size]
                _labels = _whole_labels[:-test_size]
            else:
                _ids = _whole_ids[-test_size:]
                _feats = _whole_feats[-test_size:]
                _labels = _whole_labels[-test_size:]

        elif name == 'epsilon': # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/epsilon_train.csv'),
                    delimiter=',')
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/epsilon_test.csv'),
                    delimiter=',')
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'census': # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/census_income_train.csv'),
                    delimiter=',')
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/census_income_test.csv'),
                    delimiter=',')
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'credit': # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/give_me_some_credit_train.csv'),
                    delimiter=',')
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/give_me_some_credit_test.csv'),
                    delimiter=',')
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        elif name == 'default_credit': # classification
            if train:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/default_credit_train.csv'),
                    delimiter=',')
            else:
                np_csv = np.genfromtxt(
                    os.path.join(curr_path, '../data/tabular/default_credit_test.csv'),
                    delimiter=',')
            _ids = np_csv[:, 0].astype(np.int32)
            _labels = np_csv[:, 1].astype(np.int32)
            _feats = np_csv[:, 2:]

        else:
            raise ValueError('Invalid dataset name.')

        # 2. Apply feature permutation to the train features or validate features
        if perm_option == Const.SEQUENCE:
            permuted_feats = _feats
        elif perm_option == Const.RANDOM:
            np.random.seed(seed)
            permuted_feats = _feats[:, np.random.permutation(_feats.shape[1])]
            del _feats # save memory
        elif perm_option == Const.IMPORTANCE:
            raise NotImplementedError('To be implemented...')
        else:
            raise ValueError('Invalid permutation option.')

        # 3. Split the features into active party and passive party
        num_passive_feats = int(frac * permuted_feats.shape[1])
        if role == Const.PASSIVE_NAME:
            splitted_feats = permuted_feats[:, :num_passive_feats]
            np_dataset = np.concatenate((_ids[:, np.newaxis], splitted_feats), axis=1)
        else:
            splitted_feats = permuted_feats[:, num_passive_feats:]
            np_dataset = np.concatenate((_ids[:, np.newaxis], _labels[:, np.newaxis], splitted_feats),
                                        axis=1)

        return np_dataset
'''