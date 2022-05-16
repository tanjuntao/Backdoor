import os

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from termcolor import colored

from linkefl.dataio.base import BaseDataset
from linkefl.common.const import Const


class NumpyDataset(BaseDataset):
    def __init__(self, role, abs_path=None, existing_dataset=None):
        super(NumpyDataset, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.role = role

        if existing_dataset is None:
            if abs_path is not None:
                self.np_dataset = np.genfromtxt(abs_path, delimiter=',')
            else:
                raise Exception('abs_path should not be None')
        else:
            self.np_dataset = existing_dataset

        self.has_label = True if role == Const.ACTIVE_NAME else False

    @classmethod
    def train_test_split(cls, role, whole_dataset, test_size, seed=1314):
        """Split the whole np_dataset into trainset and testset according to specific seed"""
        assert 0 < test_size < 1, 'validate size should be in range (0, 1)'

        n_train_samples = int(whole_dataset.n_samples * (1 - test_size))
        np.random.seed(seed)
        perm = np.random.permutation(whole_dataset.n_samples)
        np_trainset = whole_dataset.np_dataset[perm[:n_train_samples], :]
        np_testset = whole_dataset.np_dataset[perm[n_train_samples:], :]

        trainset = cls(role=role, existing_dataset=np_trainset)
        testset = cls(role=role, existing_dataset=np_testset)

        return trainset, testset

    @property
    def ids(self):
        np_ids = self.np_dataset[:, 0].astype(np.int32)
        list_ids = np_ids.tolist()
        return list_ids

    @property
    def features(self):
        if self.role == Const.ACTIVE_NAME:
            return self.np_dataset[:, 2:]
        else:
            return self.np_dataset[:, 1:]

    @property
    def labels(self):
        if self.role == Const.ACTIVE_NAME:
            _labels = self.np_dataset[:, 1].astype(np.int32)
            return _labels
        else:
            raise AttributeError('Passive party has no labels.')

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_samples(self):
        return self.features.shape[0]

    def describe(self):
        print(colored('Number of samples: {}'.format(self.n_samples), 'red'))
        print(colored('Number of features: {}'.format(self.n_features), 'red'))
        if self.role == Const.ACTIVE_NAME:
            n_positive = (self.labels == 1).astype(np.int32).sum()
            n_negative = self.n_samples - n_positive
            print(colored('Positive samples: Negative samples = {}:{}'
                          .format(n_positive, n_negative), 'red'))

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
        idxes = []
        all_ids = np.array(self.ids)
        for _id in intersect_ids:
            idx = np.where(all_ids == _id)[0][0]
            idxes.append(idx)
        self.np_dataset = self.np_dataset[idxes]

    def set_dataset(self, new_np_dataset):
        self.np_dataset = new_np_dataset


class BuildinNumpyDataset(NumpyDataset):
    def __init__(self, dataset_name, train, role, passive_feat_frac, feat_perm_option, seed=1314):
        assert dataset_name in Const.BUILDIN_DATASET, f"{dataset_name} is not a" \
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

        self.np_dataset = self._load_dataset(dataset_name, train, role,
                                             passive_feat_frac, feat_perm_option, seed)

    def _load_dataset(self, name, train, role, frac, perm_option, seed):
        curr_path = os.path.abspath(os.path.dirname(__file__))

        # 1. load whole dataset and split it into trainset and testset
        if name == 'cancer':
            cancer = load_breast_cancer()
            _whole_feats = cancer.data
            _whole_labels = cancer.target
            _n_samples = len(_whole_labels)
            _whole_ids = np.arange(_n_samples)
            np.random.seed(seed)
            shuffle = np.random.permutation(_n_samples)
            test_size = 0.2
            _n_train_samples = int(_n_samples * (1 - test_size))
            if train:
                _ids = _whole_ids[shuffle[:_n_train_samples]]
                _feats = _whole_feats[:_n_train_samples, :]
                _labels = _whole_labels[:_n_train_samples]
            else:
                _ids = _whole_ids[shuffle[_n_train_samples:]]
                _feats = _whole_feats[_n_train_samples:, :]
                _labels = _whole_labels[_n_train_samples:]

        elif name == 'digits':
            _whole_feats, _whole_labels = load_digits(return_X_y=True)
            _n_samples = len(_whole_labels)
            odd_idxes = np.where(_whole_labels % 2 == 1)[0]
            even_idxes = np.where(_whole_labels % 2 == 0)[0]
            _whole_labels[odd_idxes] = 1
            _whole_labels[even_idxes] = 0
            _whole_ids = np.arange(len(_whole_labels))
            np.random.seed(seed)
            shuffle = np.random.permutation(_n_samples)
            test_size = 0.2
            _n_train_samples = int(_n_samples * (1 - test_size))
            if train:
                _ids = _whole_ids[shuffle[:_n_train_samples]]
                _feats = _whole_feats[:_n_train_samples, :]
                _labels = _whole_labels[:_n_train_samples]
            else:
                _ids = _whole_ids[shuffle[_n_train_samples:]]
                _feats = _whole_feats[_n_train_samples:, :]
                _labels = _whole_labels[_n_train_samples:]

        elif name == 'epsilon':
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

        elif name == 'census':
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

        elif name == 'credit':
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

        else:
            raise ValueError('Invalid dataset name.')

        # 2. Apply feature permutation to the train features or validate features
        if perm_option == Const.SEQUENCE:
            _feats = _feats
        elif perm_option == Const.RANDOM:
            np.random.seed(seed)
            _feats = _feats[:, np.random.permutation(_feats.shape[1])]
        elif perm_option == Const.IMPORTANCE:
            raise NotImplementedError('To be implemented...')
        else:
            raise ValueError('Invalid permutation option.')

        # 3. Split the features into active party and passive party
        num_passive_feats = int(frac * _feats.shape[1])
        if role == Const.PASSIVE_NAME:
            _feats = _feats[:, :num_passive_feats]
            np_dataset = np.concatenate((_ids[:, np.newaxis], _feats), axis=1)
        else:
            _feats = _feats[:, num_passive_feats:]
            np_dataset = np.concatenate((_ids[:, np.newaxis], _labels[:, np.newaxis], _feats),
                                        axis=1)

        return np_dataset