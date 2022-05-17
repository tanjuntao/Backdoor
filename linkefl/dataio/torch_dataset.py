import os

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from termcolor import colored
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from linkefl.common.const import Const
from linkefl.dataio.base import BaseDataset


class TorchDataset(BaseDataset, Dataset):
    def __init__(self, role, abs_path=None, existing_dataset=None):
        super(TorchDataset, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.role = role

        if existing_dataset is None:
            if abs_path is not None:
                self.torch_dataset = torch.from_numpy(np.genfromtxt(abs_path, delimiter=','))
            else:
                raise Exception('data file path is not provided.')
        else:
            self.torch_dataset = existing_dataset

        self.has_label = True if role == Const.ACTIVE_NAME else False

    @classmethod
    def train_test_split(cls, role, whole_dataset, test_size, seed=1314):
        assert 0 < test_size < 1, 'test size should be in range (0, 1)'

        n_train_samples = int(whole_dataset.n_samples * (1 - test_size))
        torch.random.manual_seed(seed)
        perm = torch.randperm(whole_dataset.n_samples)
        torch_trainset = whole_dataset.torch_dataset[perm[:n_train_samples], :]
        torch_testset = whole_dataset.torch_dataset[perm[n_train_samples:], :]

        trainset = cls(role=role, existing_dataset=torch_trainset)
        testset = cls(role=role, existing_dataset=torch_testset)

        return trainset, testset

    @property
    def ids(self):
        torch_ids = self.torch_dataset[:, 0].type(torch.int32)
        list_ids = torch_ids.tolist()
        return list_ids

    @property
    def features(self):
        if self.role == Const.ACTIVE_NAME:
            return self.torch_dataset[:, 2:]
        else:
            return self.torch_dataset[:, 1:]

    @property
    def labels(self):
        if self.role == Const.ACTIVE_NAME:
            _labels = self.torch_dataset[:, 1].type(torch.int32)
            return _labels
        else:
            raise AttributeError('Passive party has no labels')

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_samples(self):
        return self.features.shape[0]

    def describe(self):
        print(colored(f"Number of samples: {self.n_samples}", 'red'))
        print(colored(f"Number of features: {self.n_features}", 'red'))
        if self.role == Const.ACTIVE_NAME:
            n_positive = (self.labels == 1).type(torch.int32).sum().item()
            n_negative = (self.labels == 0).type(torch.int32).sum().item()
            print(colored(f"Positive samples: Negative samples = "
                          f"{n_positive}:{n_negative}", 'red'))

    def filter(self, intersect_ids):
        idxes = []
        all_ids = self.torch_dataset[:, 0].type(torch.int32)
        for _id in intersect_ids:
            idx = torch.where(all_ids == _id)[0].item()
            idxes.append(idx)
        self.torch_dataset = self.torch_dataset[idxes]

    def set_dataset(self, new_torch_dataset):
        self.torch_dataset = new_torch_dataset

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.role == Const.ACTIVE_NAME:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]


class BuildinTorchDataset(TorchDataset):
    def __init__(self,
                 dataset_name,
                 role,
                 train,
                 passive_feat_frac,
                 feat_perm_option,
                 seed=1314):
        assert dataset_name in Const.BUILDIN_DATASET, f"{dataset_name} is not a buildin dataset"
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        assert 0 < passive_feat_frac < 1, 'the feature fraction of passive party' \
                                          'should be in range (0, 1)'
        assert feat_perm_option in (Const.RANDOM, Const.SEQUENCE, Const.IMPORTANCE),\
            'the feature permutation option should be among random, sequence and importance'

        self.dataset_name = dataset_name
        self.role = role
        self.train = train
        self.passive_feat_frac = passive_feat_frac
        self.feat_perm_option = feat_perm_option
        self.seed = seed

        self.torch_dataset = self._load_dataset(name=dataset_name,
                                                role=role,
                                                train=train,
                                                frac=passive_feat_frac,
                                                perm_option=feat_perm_option,
                                                seed=seed)

    def _load_dataset(self, name, role, train, frac, perm_option, seed):
        curr_path = os.path.abspath(os.path.dirname(__file__))

        if name == 'cancer':
            raise NotImplementedError('future work')

        elif name == 'digits':
            raise NotImplementedError('future work')

        elif name == 'epsilon':
            if train:
                abs_path = os.path.join(
                    curr_path,
                    '../data/tabular/epsilon_train.csv'
                )
            else:
                abs_path = os.path.join(
                    curr_path,
                    '../data/tabular/epsilon_test.csv'
                )
            torch_csv = torch.from_numpy(np.genfromtxt(abs_path, delimiter=','))
            _ids = torch_csv[:, 0].type(torch.int32)
            _labels = torch_csv[:, 1].type(torch.int32)
            _feats = torch_csv[:, 2:]

        elif name == 'census':
            if train:
                abs_path = os.path.join(
                    curr_path,
                    '../data/tabular/census_income_train.csv'
                )
            else:
                abs_path = os.path.join(
                    curr_path,
                    '../data/tabular/census_income_test.csv'
                )
            torch_csv = torch.from_numpy(np.genfromtxt(abs_path, delimiter=','))
            _ids = torch_csv[:, 0].type(torch.int32)
            _labels = torch_csv[:, 1].type(torch.int32)
            _feats = torch_csv[:, 2:]

        elif name == 'credit':
            if train:
                abs_path = os.path.join(
                    curr_path,
                    '../data/tabular/give_me_some_credit_train.csv'
                )
            else:
                abs_path = os.path.join(
                    curr_path,
                    '../data/tabular/give_me_some_credit_test.csv'
                )
            torch_csv = torch.from_numpy(np.genfromtxt(abs_path, delimiter=','))
            _ids = torch_csv[:, 0].type(torch.int32)
            _labels = torch_csv[:, 1].type(torch.int32)
            _feats = torch_csv[:, 2:]

        elif name == 'mnist':
            buildin_dataset = datasets.MNIST(root='data',
                                             train=train,
                                             download=True,
                                             transform=ToTensor())
            n_samples = buildin_dataset.data.shape[0]
            _ids = torch.arange(n_samples)
            _labels = buildin_dataset.targets
            _feats = buildin_dataset.view(n_samples, -1)  # shape: n_samples, 28*28

        elif name == 'fashion_mnist':
            buildin_dataset = datasets.FashionMNIST(root='data',
                                                    train=train,
                                                    download=True,
                                                    transform=ToTensor())
            n_samples = buildin_dataset.data.shape[0]
            _ids = torch.arange(n_samples)
            _labels = buildin_dataset.targets
            _feats = buildin_dataset.view(n_samples, -1)  # shape: n_samples, 28*28

        elif name == 'svhn':
            split = 'train' if train else 'validate'
            buildin_dataset = datasets.SVHN(root='data',
                                            split=split,
                                            download=True,
                                            transform=ToTensor())
            n_samples = buildin_dataset.data.shape[0]
            _ids = torch.arange(n_samples)
            _labels = buildin_dataset.labels
            _feats = buildin_dataset.view(n_samples, -1)

        else:
            raise ValueError('Invalid dataset name.')

        if perm_option == Const.SEQUENCE:
            _feats = _feats
        elif perm_option == Const.RANDOM:
            torch.random.manual_seed(seed)
            n_feats = _feats.shape[1]
            _feats = _feats[:, torch.randperm(n_feats)]
        elif perm_option == Const.IMPORTANCE:
            raise NotImplementedError('future work')

        num_passive_feats = int(frac * _feats.shape[1])
        if role == Const.PASSIVE_NAME:
            _feats = _feats[:, :num_passive_feats]
            torch_dataset = torch.cat((torch.unsqueeze(_ids, 1), _feats), dim=1)
        else:
            _feats = _feats[:, num_passive_feats:]
            torch_dataset = torch.cat((torch.unsqueeze(_ids, 1),
                                       torch.unsqueeze(_labels, 1),
                                       _feats),
                                      dim=1)

        return torch_dataset



