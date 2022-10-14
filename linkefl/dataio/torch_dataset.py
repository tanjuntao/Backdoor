import os
from urllib.error import URLError

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
from termcolor import colored
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt

from linkefl.common.const import Const
from linkefl.dataio.base import BaseDataset
from linkefl.util import urlretrive


class TorchDataset(BaseDataset, Dataset):
    def __init__(self, role, abs_path=None, transform=None, existing_dataset=None):
        super(TorchDataset, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        self.role = role

        # if existing_dataset is None:
        #     if abs_path is not None:
        #         self._torch_dataset = torch.from_numpy(
        #             np.genfromtxt(abs_path, dtype=np.float32, delimiter=','))
        #         # self._torch_dataset = pd.read_csv(abs_path, delimiter=',', header=None)
        #     else:
        #         raise Exception('data file path is not provided.')
        # else:
        #     self.set_dataset(existing_dataset)

        torch_data = TorchDataset._load_csv_dataset(abs_path, existing_dataset)
        self.set_dataset(torch_data)

        # if transform is not None:
        #     # self._torch_dataset = transform(self._torch_dataset)
        #     self._torch_dataset = transform(self._torch_dataset, role=role)
        if transform is not None:
            temp = transform(self._torch_dataset, role=role)
            if isinstance(temp, tuple):
                self._torch_dataset, self.bin_split = temp
            else:
                self._torch_dataset = temp
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

    @classmethod
    def buildin_dataset(cls, dataset_name, role, root, train, passive_feat_frac,
                        feat_perm_option, download=False, transform=None, seed=1314):
        def _check_params():
            assert dataset_name in Const.BUILDIN_DATASETS, f"{dataset_name} is not a buildin dataset"
            assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
            assert 0 < passive_feat_frac < 1, 'the feature fraction of passive party' \
                                              'should be in range (0, 1)'
            assert feat_perm_option in (Const.RANDOM, Const.SEQUENCE, Const.IMPORTANCE), \
                'the feature permutation option should be among random, sequence and importance'

        # function body
        _check_params()
        torch_dataset = TorchDataset._load_buildin_dataset(
            role=role, name=dataset_name, root=root, train=train,
            download=download, frac=passive_feat_frac, perm_option=feat_perm_option,
            seed=seed
        )
        return cls(
            role=role,
            transform=transform,
            existing_dataset=torch_dataset
        )

    @staticmethod
    def _load_csv_dataset(path, existing_dataset=None):
        if existing_dataset is None:
            if path is not None:
                torch_dataset = torch.from_numpy(
                    np.genfromtxt(path, dtype=np.float32, delimiter=','))
                # self._torch_dataset = pd.read_csv(abs_path, delimiter=',', header=None)
            else:
                raise Exception('data file path is not provided.')
        else:
            torch_dataset = existing_dataset

        return torch_dataset

    @staticmethod
    def _load_buildin_dataset(role, name, root, train, download, frac, perm_option, seed):
        def _check_exists(dataset_name, root_, train_, resources_):
            if train_:
                filename_ = resources_[dataset_name][0]
            else:
                filename_ = resources_[dataset_name][1]
            return os.path.exists(os.path.join(root_, filename_))

        # 1. Load dataset
        if name not in ('mnist', 'fashion_mnist', 'svhn'): # CSV datasets
            resources = {
                "cancer": ("cancer-train.csv", "cancer-test.csv"),
                "digits": ("digits-train.csv", "digits-test.csv"),
                "diabetes": ("diabetes-train.csv", "diabetes-test.csv"),
                "iris": ("iris-train.csv", "iris-test.csv"),
                "wine": ("wine-train.csv", "wine-test.csv"),
                "epsilon": ("epsilon-train.csv", "epsilon-test.csv"),
                "census": ("census-train.csv", "census-test.csv"),
                "credit": ("credit-train.csv", "credit-test.csv"),
                "default_credit": (
                "default-credit-train.csv", "default-credit-test.csv"),
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
                        raise RuntimeError(
                            'Failed to download {} with error message: {}'
                            .format(full_url, error))
                    print('Done!')
            if not _check_exists(name, root, train, resources):
                raise RuntimeError('Dataset not found. You can use download=True to get it.')

            if train:
                fpath = os.path.join(root, resources[name][0])
            else:
                fpath = os.path.join(root, resources[name][1])
            torch_csv = torch.from_numpy(np.genfromtxt(fpath, dtype=np.float32, delimiter=','))
            _ids = torch_csv[:, 0].type(torch.int32)
            _labels = torch_csv[:, 1].type(torch.int32)
            _feats = torch_csv[:, 2:]

        else: # PyTorch datasets
            if name == 'mnist':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                buildin_dataset = datasets.MNIST(root=root,
                                                 train=train,
                                                 download=download,
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
                    image, _ = buildin_dataset[i]  # image shape: [1, 28, 28]
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
                buildin_dataset = datasets.FashionMNIST(root=root,
                                                        train=train,
                                                        download=download,
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
                buildin_dataset = datasets.SVHN(root=root,
                                                split=split,
                                                download=download,
                                                transform=transforms.ToTensor())
                n_samples = buildin_dataset.data.shape[0]
                _ids = torch.arange(n_samples)
                _labels = buildin_dataset.labels
                _feats = buildin_dataset.data.view(n_samples, -1)

            else:
                raise ValueError('not supported right now')

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

    @property
    def ids(self):  # read only
        # avoid re-computing on each function call
        if not hasattr(self, '_ids'):
            torch_ids = self._torch_dataset[:, 0].type(torch.int32)
            py_ids = [_id.item() for _id in torch_ids]
            setattr(self, '_ids', py_ids)
        return getattr(self, '_ids')

    @property
    def features(self):  # read only
        if not hasattr(self, '_features'):
            if self.role == Const.ACTIVE_NAME:
                setattr(self, '_features', self._torch_dataset[:, 2:])
            else:
                setattr(self, '_features', self._torch_dataset[:, 1:])
        return getattr(self, '_features')

    @property
    def labels(self):  # read only
        if self.role == Const.PASSIVE_NAME:
            raise AttributeError('Passive party has no labels.')

        if not hasattr(self, '_labels'):
            # the type of labels should be torch.long, otherwise,
            # the RuntimeError: expected scalar type Long but found Int will be raised
            setattr(self, '_labels', self._torch_dataset[:, 1].type(torch.long))
        return getattr(self, '_labels')

    @property
    def n_features(self):  # read only
        return self.features.shape[1]

    @property
    def n_samples(self):  # read only
        return self.features.shape[0]

    def describe(self):
        print(colored(f"Number of samples: {self.n_samples}", 'red'))
        print(colored(f"Number of features: {self.n_features}", 'red'))
        if self.role == Const.ACTIVE_NAME:
            n_positive = (self.labels == 1).type(torch.int32).sum().item()
            n_negative = self.n_samples - n_positive
            print(colored(f"Positive samples: Negative samples = "
                          f"{n_positive}:{n_negative}", 'red'))
        print()

        # Output of statistical values of the data set.
        pd.set_option('display.max_columns', None)
        dataset = pd.DataFrame(self._torch_dataset)
        if self.role == Const.ACTIVE_NAME:
            dataset.rename(columns={0: 'id', 1: 'lable'}, inplace=True)
            for i in range(self.n_features):
                dataset.rename(columns={i + 2: 'fea' + str(i + 1)}, inplace=True)
        elif self.role == Const.PASSIVE_NAME:
            dataset.rename(columns={0: 'id'}, inplace=True)
            for i in range(self.n_features):
                dataset.rename(columns={i + 1: 'fea' + str(i + 1)}, inplace=True)
        data_cols = dataset.columns.values.tolist()

        print(colored('The first 5 rows and the last 5 rows of the dataset are as follows:', 'red'))
        print(pd.concat([dataset.head(), dataset.tail()]))
        print()

        print(colored(
            'The information about the dataset including the index dtype and columns, non-null values and memory usage are as follows:',
            'red'))
        dataset.info()
        print()

        print(colored(
            'The descriptive statistics include those that summarize the central tendency, dispersion and shape of the dataset’s distribution, excluding NaN values are as follows:',
            'red'))
        num_unique_data = np.array(dataset[data_cols].nunique().values)
        num_unique = pd.DataFrame(data=num_unique_data.reshape((1, -1)), index=['unique'], columns=data_cols)

        print(pd.concat([dataset.describe(), num_unique]))
        print()

        # Output the distribution for the data label.
        if self.role == Const.ACTIVE_NAME:
            dis_label = pd.DataFrame(data=self.labels.reshape((-1, 1)), columns=['label'])
            # 图中虚线是核密度曲线（类似于概率密度）
            sns.histplot(dis_label, kde=True, linewidth=0)
            plt.show()

    def filter(self, intersect_ids):
        # convert intersection ids to Python list
        if type(intersect_ids) == torch.Tensor:
            intersect_ids = intersect_ids.tolist()
        if type(intersect_ids) == np.ndarray:
            intersect_ids = intersect_ids.tolist()
        if type(intersect_ids) == list:
            pass

        idxes = []
        all_ids = torch.tensor(self.ids)
        for _id in intersect_ids:
            idx = torch.where(all_ids == _id)[0].item()
            idxes.append(idx)
        new_torch_dataset = self._torch_dataset[idxes]
        self.set_dataset(new_torch_dataset)

    def get_dataset(self):
        return self._torch_dataset

    def set_dataset(self, new_torch_dataset):
        assert isinstance(new_torch_dataset, torch.Tensor), \
            "new_torch_dataset should be an instance of torch.Tensor"

        # mush delete old properties to save memory
        if hasattr(self, '_torch_dataset'):
            del self._torch_dataset
        if hasattr(self, '_ids'):
            del self._ids
        if hasattr(self, '_features'):
            del self._features
        if hasattr(self, '_labels'):
            del self._labels

        # update new property
        self._torch_dataset = new_torch_dataset

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.role == Const.ACTIVE_NAME:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

'''
class BuildinTorchDataset(TorchDataset):
    def __init__(self,
                 dataset_name,
                 role,
                 train,
                 passive_feat_frac,
                 feat_perm_option,
                 transform=None,
                 seed=1314
                 ):
        assert dataset_name in Const.BUILDIN_DATASETS, f"{dataset_name} is not a buildin dataset"
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
        assert 0 < passive_feat_frac < 1, 'the feature fraction of passive party' \
                                          'should be in range (0, 1)'
        assert feat_perm_option in (Const.RANDOM, Const.SEQUENCE, Const.IMPORTANCE), \
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
        if transform is not None:
            self._torch_dataset = transform(self._torch_dataset)
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

        elif name == 'default_credit':
            if train:
                abs_path = os.path.join(
                    curr_path,
                    '../data/tabular/default_credit_train.csv'
                )
            else:
                abs_path = os.path.join(
                    curr_path,
                    '../data/tabular/default_credit_test.csv'
                )
            torch_csv = torch.from_numpy(np.genfromtxt(abs_path, dtype=np.float32, delimiter=','))
            _ids = torch_csv[:, 0].type(torch.int32)
            _labels = torch_csv[:, 1].type(torch.int32)
            _feats = torch_csv[:, 2:]

        elif name == 'criteo':
            if train:
                abs_path = os.path.join(
                    curr_path,
                    '../vfl/linear/data/tabular/criteo-train.csv'
                )
            else:
                abs_path = os.path.join(
                    curr_path,
                    '../vfl/linear/data/tabular/criteo-test.csv'
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
                image, _ = buildin_dataset[i]  # image shape: [1, 28, 28]
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
    '''
