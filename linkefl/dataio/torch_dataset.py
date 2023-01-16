import random
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from linkefl.base import BaseTransformComponent
from linkefl.common.const import Const
from linkefl.dataio.common_dataset import CommonDataset


class TorchDataset(CommonDataset, Dataset):
    def __init__(self,
                 role: str,
                 raw_dataset: Union[np.ndarray, torch.Tensor],
                 header: list,
                 dataset_type: str,
                 transform: BaseTransformComponent = None,
                 header_type =  None,
    ):
        if isinstance(raw_dataset, np.ndarray):
            # PyTorch forward() function expects tensor type of Float rather Double,
            raw_dataset = raw_dataset.astype(np.float32)
            raw_dataset = torch.from_numpy(raw_dataset) # very important

        # since there is no super() in CommonDataset's __init__() constructor,
        # python's MRO(method resolution order) will not be triggered, and only
        # CommonDataset's __init__() constructor is being called.
        # ref: https://stackoverflow.com/a/16310777/8418540
        super(TorchDataset, self).__init__(
            role=role,
            raw_dataset=raw_dataset,
            header=header,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type
        )

    @staticmethod
    def _load_buildin_dataset(role, name,
                              root, train, download,
                              frac, perm_option,
                              seed=None
    ):
        if name not in Const.PYTORCH_DATASET:
            # the following answer shows how to call staticmethod in superclass:
            # ref: https://stackoverflow.com/a/26807879/8418540
            np_dataset, header = super(TorchDataset, TorchDataset)._load_buildin_dataset(
                role=role, name=name,
                root=root, train=train, download=download,
                frac=frac, perm_option=perm_option,
                seed=seed
            )
            return np_dataset, header

        # 1. Load PyTorch datasets
        from torchvision import datasets, transforms
        from tqdm import trange

        from linkefl.feature import cal_importance_ranking

        if name == 'tab_mnist':
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
            for i in trange(n_samples):
                # this will trigger the __getitem__ method and the images will be
                # normalized. The return type is a tuple composed of (PILImage, label)
                image, _ = buildin_dataset[i]  # image shape: [1, 28, 28]
                img = image.view(1, -1)
                imgs.append(img)
            _feats = torch.Tensor(n_samples, n_features)
            _feats_header = ['x{}'.format(i) for i in range(n_features)]
            torch.cat(imgs, out=_feats)

            # solution 2: use tensor.index_copy_()
            # _feats = torch.zeros(n_samples, n_features)
            # for i in range(n_samples):
            #     image, _ = buildin_dataset[i]
            #     img = image.view(1, -1)
            #     _feats.index_copy_(0, torch.tensor([i], dtype=torch.long), img)

        elif name == 'tab_fashion_mnist':
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
            for i in trange(n_samples):
                # this will trigger the __getitem__ method
                # the return type is a tuple composed of (PILImage, label)
                image, _ = buildin_dataset[i]  # image shape: [1, 28, 28]
                img = image.view(1, -1)
                imgs.append(img)
            _feats = torch.Tensor(n_samples, n_features)
            _feats_header = ['x{}'.format(i) for i in range(n_features)]
            torch.cat(imgs, out=_feats)

        else:
            raise ValueError('not supported right now')

        # 2. Apply feature permutation to the train features or validate features
        if perm_option == Const.SEQUENCE:
            _feats = _feats
            _feats_header = _feats_header
        elif perm_option == Const.RANDOM:
            if seed is not None:
                random.seed(seed)
            perm = list(range(_feats.shape[1]))
            random.shuffle(perm)
            _feats = _feats[:, perm]
            _feats_header = np.array(_feats_header)[perm].tolist()
        elif perm_option == Const.IMPORTANCE:
            rankings_np = cal_importance_ranking(name, _feats.numpy(), _labels.numpy())
            rankings = torch.from_numpy(rankings_np)
            _feats = _feats[:, rankings]
            _feats_header = np.array(_feats_header)[rankings_np].tolist()

        # 3. Split the features into active party and passive party
        num_passive_feats = int(frac * _feats.shape[1])
        if role == Const.PASSIVE_NAME:
            _feats = _feats[:, :num_passive_feats]
            header = ['id'] + _feats_header
            torch_dataset = torch.cat(
                (torch.unsqueeze(_ids, 1), _feats),
                dim=1
            )
        else:
            _feats = _feats[:, num_passive_feats:]
            header = ['id'] + ['y'] + _feats_header
            torch_dataset = torch.cat(
                (torch.unsqueeze(_ids, 1), torch.unsqueeze(_labels, 1), _feats),
                dim=1
            )

        return torch_dataset, header

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.role == Const.ACTIVE_NAME:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]


class MediaDataset(TorchDataset, Dataset):
    def __init__(self,
                 role,
                 dataset_name,
                 root,
                 train,
                 download=False,
    ):
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), "invalid role name"
        assert dataset_name in ("cifar10", "mnist", "fashion_mnist"), "not supported dataset"

        self.role = role
        self._prepare_data(
            role=role,
            name=dataset_name,
            root=root,
            train=train,
            download=download
        )

    def __len__(self):
        return getattr(self, '_features').shape[0]

    def __getitem__(self, idx):
        if self.role == Const.ACTIVE_NAME:
            return getattr(self, '_features')[idx], getattr(self, '_labels')[idx]
        else:
            return getattr(self, '_features')[idx]

    @property
    def labels(self):
        return getattr(self, '_labels')

    def _prepare_data(self, role, name, root, train, download):
        from torchvision import datasets, transforms
        from tqdm import trange

        if name == 'cifar10':
            if train:
                transform = transforms.Compose([
                    transforms.Resize((64, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((64, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            buildin_dataset = datasets.CIFAR10(
                root=root,
                train=train,
                download=download,
                transform=transform
            )

            n_samples = len(buildin_dataset)
            imgs = []
            for i in trange(n_samples):
                image, _ = buildin_dataset[i]
                if role == Const.PASSIVE_NAME:
                    image = image[:, :32, :]  # the first half
                else:
                    image = image[:, 32:, :]  # the second half
                imgs.append(image)

        elif name == 'mnist':
            if train:
                transform = transforms.Compose([
                    transforms.Resize((64, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((64, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
            buildin_dataset = datasets.MNIST(
                root=root,
                train=train,
                download=download,
                transform=transform
            )

            n_samples = len(buildin_dataset)
            imgs = []
            for i in trange(n_samples):
                image, _ = buildin_dataset[i]
                if role == Const.PASSIVE_NAME:
                    image = image[:, :32, :]  # the first half
                else:
                    image = image[:, 32:, :]  # the second half
                imgs.append(image)

        else:
            raise ValueError("not suported now.")

        _feats = torch.stack(imgs)  # stack() will create a new dimension
        _labels = torch.tensor(buildin_dataset.targets, dtype=torch.long)
        if role == Const.ACTIVE_NAME:
            setattr(self, '_features', _feats)
            setattr(self, '_labels', _labels)
        else:
            setattr(self, '_features', _feats)


if __name__ == '__main__':
    # from torchvision import datasets, transforms
    #
    # _transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    # _buildin_dataset = datasets.CIFAR10(root='data',
    #                                         train=True,
    #                                         download=True,
    #                                         transform=_transform)
    # _image, label = _buildin_dataset[0]
    # print(type(_image), type(label))
    # print(_image.shape)
    # print(_buildin_dataset.data.shape)
    # print(_buildin_dataset.data[0].shape)

    cifar_dataset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name='cifar10',
        root='data',
        train=False,
        download=True
    )
    _image, _label = cifar_dataset[0]
    print(type(_image), type(_label))
    print(_image.shape)