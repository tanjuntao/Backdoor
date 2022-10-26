from typing import Union

import numpy as np
import torch

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset


class TorchDataset(NumpyDataset):
    def __init__(self, role, dataset: Union[np.ndarray, torch.Tensor], dataset_type, transform=None):
        if isinstance(dataset, np.ndarray):
            dataset = torch.from_numpy(dataset)

        super().__init__(role, dataset, dataset_type, transform)

    @staticmethod
    def _load_buildin_dataset(role, name, root, train, frac, perm_option, download, seed=1314):
        if name not in ('mnist', 'fashion_mnist', 'svhn'):
            # Load CSV datasets
            np_dataset = super()._load_buildin_dataset(role, name, root, train, frac, perm_option, download, seed)
            return np_dataset

        # 1. Load PyTorch datasets
        from torchvision import datasets, transforms

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

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.role == Const.ACTIVE_NAME:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]
