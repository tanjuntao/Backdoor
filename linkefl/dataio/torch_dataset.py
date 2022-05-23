import os

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from termcolor import colored
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms

from linkefl.common.const import Const
from linkefl.dataio.base import BaseDataset


class TorchDataset(BaseDataset, Dataset):
    def __init__(self, role, abs_path=None, existing_dataset=None):
        super(TorchDataset, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.role = role

        if existing_dataset is None:
            if abs_path is not None:
                self._torch_dataset = torch.from_numpy(np.genfromtxt(abs_path, delimiter=','))
            else:
                raise Exception('data file path is not provided.')
        else:
            self.set_dataset(existing_dataset)

        self.has_label = True if role == Const.ACTIVE_NAME else False

    @classmethod
    def train_test_split(cls, role, whole_dataset, test_size, seed=1314):
        assert isinstance(whole_dataset, TorchDataset), 'whole_dataset should be' \
                                                        'an instance of TorchDataset'
        assert 0 < test_size < 1, 'test size should be in range (0, 1)'

        n_train_samples = int(whole_dataset.n_samples * (1 - test_size))
        torch.random.manual_seed(seed)
        perm = torch.randperm(whole_dataset.n_samples)
        torch_trainset = whole_dataset.get_dataset()[perm[:n_train_samples], :]
        torch_testset = whole_dataset.get_dataset()[perm[n_train_samples:], :]

        trainset = cls(role=role, existing_dataset=torch_trainset)
        testset = cls(role=role, existing_dataset=torch_testset)

        return trainset, testset

    @property
    def ids(self): # read only
        # avoid re-computing on each function call
        if not hasattr(self, '_ids'):
            torch_ids = self._torch_dataset[:, 0].type(torch.int32)
            self._ids = torch_ids
        return self._ids

    @property
    def features(self): # read only
        if not hasattr(self, '_features'):
            if self.role == Const.ACTIVE_NAME:
                self._features = self._torch_dataset[:, 2:]
            else:
                self._features = self._torch_dataset[:, 1:]
        return self._features

    @property
    def labels(self): # read only
        if self.role == Const.PASSIVE_NAME:
            raise AttributeError('Passive party has no labels.')

        if not hasattr(self, '_labels'):
            # the type of labels should be torch.long, otherwise,
            # the RuntimeError: expected scalar type Long but found Int will be raised
            self._labels = self._torch_dataset[:, 1].type(torch.long)
        return self._labels

    @property
    def n_features(self): # read only
        return self.features.shape[1]

    @property
    def n_samples(self): # read only
        return self.features.shape[0]

    def describe(self):
        print(colored(f"Number of samples: {self.n_samples}", 'red'))
        print(colored(f"Number of features: {self.n_features}", 'red'))
        if self.role == Const.ACTIVE_NAME:
            n_positive = (self.labels == 1).type(torch.int32).sum().item()
            n_negative = self.n_samples - n_positive
            print(colored(f"Positive samples: Negative samples = "
                          f"{n_positive}:{n_negative}", 'red'))

    def filter(self, intersect_ids):
        # convert intersection ids to Python list
        if type(intersect_ids) == torch.Tensor:
            intersect_ids = intersect_ids.tolist()
        if type(intersect_ids) == np.ndarray:
            intersect_ids = intersect_ids.tolist()
        if type(intersect_ids) == list:
            pass

        idxes = []
        all_ids = self.ids
        for _id in intersect_ids:
            idx = torch.where(all_ids == _id)[0].item()
            idxes.append(idx)
        self._torch_dataset = self._torch_dataset[idxes]

    def get_dataset(self):
        return self._torch_dataset

    def set_dataset(self, new_torch_dataset):
        assert isinstance(new_torch_dataset, torch.Tensor),\
            "new_torch_dataset should be an instance of torch.Tensor"
        self._torch_dataset = new_torch_dataset

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

        self._torch_dataset = self._load_dataset(name=dataset_name,
                                                role=role,
                                                train=train,
                                                frac=passive_feat_frac,
                                                perm_option=feat_perm_option,
                                                seed=seed)
        self.has_label = True if role == Const.ACTIVE_NAME else False

    def _load_dataset(self, name, role, train, frac, perm_option, seed):
        curr_path = os.path.abspath(os.path.dirname(__file__))

        # 1. load whole dataset and split it into trainset and testset
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
            torch_csv = torch.from_numpy(np.genfromtxt(abs_path, dtype=np.float32, delimiter=','))
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
            # PyTorch forward() function expects tensor type of Float rather Double
            # so the dtype should be specified to np.float32
            torch_csv = torch.from_numpy(np.genfromtxt(abs_path, dtype=np.float32, delimiter=','))
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
            torch_csv = torch.from_numpy(np.genfromtxt(abs_path, dtype=np.float32, delimiter=','))
            _ids = torch_csv[:, 0].type(torch.int32)
            _labels = torch_csv[:, 1].type(torch.int32)
            _feats = torch_csv[:, 2:]

        elif name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            buildin_dataset = datasets.MNIST(root='data',
                                             train=train,
                                             download=True,
                                             transform=transform)
            n_samples = len(buildin_dataset)
            n_features = 28 * 28
            _ids = torch.arange(n_samples)
            _labels = buildin_dataset.targets
            # A notation about the __getitem__() method of PyTorch Datasets:
            # The raw images are normalized within the __getitem__ method rather
            # than the __init__ method, which means that if we call
            # buildin_dataset.data[i] we will get the
            # i-th PILImage with pixel range of [0, 255], but if we call
            # buildin_dataset[i] we will get a tuple of (Tensor, label) where Tensor
            # is a PyTorch tensor with normalized pixel in the range (0, 1).
            # So here we first normalized the raw PILImage and then store it in
            # the _feats attribute.
            # ref:https://stackoverflow.com/questions/66821250/pytorch-totensor-scaling-to-0-1-discrepancy

            # The execution time of the following two solutions is almost same,
            # but solution 1 is a bit faster which is used as default solution.

            # solution 1: use torch.cat()
            imgs = []
            for i in range(n_samples):
                # this will trigger the __getitem__ method and the images will be
                # normalized. The return type is a tuple composed of (PILImage, label)
                image, _ = buildin_dataset[i] # image shape: [1, 28, 28]
                img = image.view(1, -1)
                imgs.append(img)
            _feats = torch.Tensor(n_samples, n_features)
            torch.cat(imgs, out=_feats)

            # solution 2: use tensor.index_copy_()
            # _feats = torch.zeros(n_samples, n_features)
            # for i in range(n_samples):
            #     image, _ = buildin_dataset[i]
            #     img = image.view(1, -1)
            #     _feats.index_copy_(0, torch.tensor([i], dtype=torch.long), img)

        elif name == 'fashion_mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            buildin_dataset = datasets.FashionMNIST(root='data',
                                                    train=train,
                                                    download=True,
                                                    transform=transform)
            n_samples = buildin_dataset.data.shape[0]
            n_features = 28 * 28
            _ids = torch.arange(n_samples)
            _labels = buildin_dataset.targets

            imgs = []
            for i in range(n_samples):
                # this will trigger the __getitem__ method
                # the return type is a tuple composed of (PILImage, label)
                image, _ = buildin_dataset[i]  # image shape: [1, 28, 28]
                img = image.view(1, -1)
                imgs.append(img)
            _feats = torch.Tensor(n_samples, n_features)
            torch.cat(imgs, out=_feats)

        elif name == 'svhn':
            # TODO: add SVHN specific transforms here
            split = 'train' if train else 'test'
            buildin_dataset = datasets.SVHN(root='data',
                                            split=split,
                                            download=True,
                                            transform=transforms.ToTensor())
            n_samples = buildin_dataset.data.shape[0]
            _ids = torch.arange(n_samples)
            _labels = buildin_dataset.labels
            _feats = buildin_dataset.data.view(n_samples, -1)

        else:
            raise ValueError('Invalid dataset name.')

        # 2. Apply feature permutation to the train features or validate features
        if perm_option == Const.SEQUENCE:
            _feats = _feats
        elif perm_option == Const.RANDOM:
            torch.random.manual_seed(seed)
            n_feats = _feats.shape[1]
            _feats = _feats[:, torch.randperm(n_feats)]
        elif perm_option == Const.IMPORTANCE:
            raise NotImplementedError('future work')

        # 3. Split the features into active party and passive party
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



