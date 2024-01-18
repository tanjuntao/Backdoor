import random
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from linkefl.base import BaseTransformComponent
from linkefl.common.const import Const
from linkefl.dataio.common_dataset import CommonDataset


class TorchDataset(CommonDataset, Dataset):
    def __init__(
        self,
        *,
        role: str,
        raw_dataset: Union[np.ndarray, torch.Tensor],
        header: List[str],
        dataset_type: str,
        transform: Optional[BaseTransformComponent] = None,
        header_type: Optional[List[str]] = None,
    ):
        if isinstance(raw_dataset, np.ndarray):
            # PyTorch forward() function expects tensor type of Float rather Double,
            raw_dataset = raw_dataset.astype(np.float32)
            raw_dataset = torch.from_numpy(raw_dataset)  # very important

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
            header_type=header_type,
        )

    @staticmethod
    def _load_buildin_dataset(
        role, name, root, train, download, frac, perm_option, seed=None
    ):
        if name not in Const.PYTORCH_DATASET:
            # the following answer shows how to call staticmethod in superclass:
            # ref: https://stackoverflow.com/a/26807879/8418540
            np_dataset, header = super(
                TorchDataset, TorchDataset
            )._load_buildin_dataset(
                role=role,
                name=name,
                root=root,
                train=train,
                download=download,
                frac=frac,
                perm_option=perm_option,
                seed=seed,
            )
            return np_dataset, header

        # 1. Load PyTorch datasets
        from torchvision import datasets, transforms
        from tqdm import trange

        from linkefl.feature import cal_importance_ranking

        if name == "tab_mnist":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            buildin_dataset = datasets.MNIST(
                root=root, train=train, download=download, transform=transform
            )
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
            _feats_header = ["x{}".format(i) for i in range(n_features)]
            torch.cat(imgs, out=_feats)

            # solution 2: use tensor.index_copy_()
            # _feats = torch.zeros(n_samples, n_features)
            # for i in range(n_samples):
            #     image, _ = buildin_dataset[i]
            #     img = image.view(1, -1)
            #     _feats.index_copy_(0, torch.tensor([i], dtype=torch.long), img)

        elif name == "tab_fashion_mnist":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            buildin_dataset = datasets.FashionMNIST(
                root=root, train=train, download=download, transform=transform
            )
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
            _feats_header = ["x{}".format(i) for i in range(n_features)]
            torch.cat(imgs, out=_feats)

        else:
            raise ValueError("not supported right now")

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
            header = ["id"] + _feats_header
            torch_dataset = torch.cat((torch.unsqueeze(_ids, 1), _feats), dim=1)
        else:
            _feats = _feats[:, num_passive_feats:]
            header = ["id"] + ["y"] + _feats_header
            torch_dataset = torch.cat(
                (torch.unsqueeze(_ids, 1), torch.unsqueeze(_labels, 1), _feats), dim=1
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
    def __init__(
        self,
        *,
        role: str,
        dataset_name: str,
        root: str,
        train: bool,
        download: bool = False,
        fine_tune=False,
        fine_tune_per_class=32,
        active_full_image=False,
        world_size=2,
        rank=0,
    ):
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), "invalid role name"
        assert dataset_name in (
            "cifar10",
            "cifar100",
            "cinic10",
            "mnist",
            "fashion_mnist",
            "svhn",
        ), f"{dataset_name} is not supported right now."

        self.role = role
        self.dataset_name = dataset_name
        self.fine_tune = fine_tune
        self.fine_tune_per_class = fine_tune_per_class
        self.active_full_image = active_full_image
        self.world_size = world_size
        self.rank = rank
        buildin_dataset, _labels = self._prepare_data(
            name=dataset_name, root=root, train=train, download=download
        )
        self.buildin_dataset = buildin_dataset
        setattr(self, "_labels", _labels)
        self.seed_maps = {idx: 0 for idx in range(len(self.buildin_dataset))}

    @property
    def labels(self):
        return getattr(self, "_labels")

    def __len__(self):
        return len(self.buildin_dataset)

    def __getitem__(self, idx):
        transform_seed = self.seed_maps[idx]
        torch.manual_seed(transform_seed)  # set seed for torchvision transform
        self.seed_maps[idx] += 1  # change seed for next epoch
        image, label = self.buildin_dataset[idx]  # torchvision transform is done here
        if self.dataset_name == "cinic10":
            label = int(label)

        # two party
        if self.world_size == 2:
            if self.role == Const.ACTIVE_NAME:
                if self.active_full_image:
                    return image, label  # full image
                else:
                    return image[:, :16, :], label  # first half image
            else:
                return image[:, 16:, :]  # second half

        # four party
        elif self.world_size == 4:
            if self.rank == 0:
                return image[:, :16, :16], label
            elif self.rank == 1:
                return image[:, :16, 16:]
            elif self.rank == 2:
                return image[:, 16:, :16]
            elif self.rank == 3:
                return image[:, 16:, 16:]
            else:
                raise ValueError(f"invalid rank: {self.rank}")

    def _prepare_data(self, name, root, train, download):
        from torchvision import datasets, transforms

        # prepare transforms and load buildin dataset
        if name in ("cifar10", "cifar100"):
            if name == "cifar10":
                mean = (0.4914, 0.4822, 0.4465)
                std = (0.2023, 0.1994, 0.2010)
                constructor = datasets.CIFAR10
            else:
                mean = (0.5071, 0.4865, 0.4409)
                std = (0.2673, 0.2564, 0.2762)
                constructor = datasets.CIFAR100
            if train:
                transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
            buildin_dataset = constructor(
                root=root, train=train, download=download, transform=transform
            )
            # """
            if self.fine_tune:
                if train:
                    import numpy as np

                    targets = np.array(buildin_dataset.targets)
                    new_data, new_target = [], []
                    n_classes = 10 if name == "cifar10" else 100
                    for label in range(n_classes):
                        curr_data = buildin_dataset.data[targets == label][
                            : self.fine_tune_per_class
                        ]
                        new_data.append(curr_data)
                        new_target.extend([label] * self.fine_tune_per_class)
                    buildin_dataset.data = np.vstack(new_data)
                    buildin_dataset.targets = new_target
                    # import random
                    # random.seed(0)
                    # perm = list(range(per_class * 10))
                    # random.shuffle(perm)
                    # buildin_dataset.data = np.array(new_data)[perm]
                    # buildin_dataset.targets = np.array(new_target)[perm].tolist()
            # """
            _labels = torch.tensor(buildin_dataset.targets, dtype=torch.long)

        elif name in ("mnist", "fashion_mnist"):
            if train:
                transform = transforms.Compose(
                    [
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                )
            if name == "mnist":
                buildin_dataset = datasets.MNIST(
                    root=root, train=train, download=download, transform=transform
                )
            else:
                buildin_dataset = datasets.FashionMNIST(
                    root=root, train=train, download=download, transform=transform
                )
            # """
            if self.fine_tune:
                if train:
                    import numpy as np

                    targets = buildin_dataset.targets
                    new_data, new_target = [], []
                    for label in range(10):
                        curr_data = buildin_dataset.data[targets == label][
                            : self.fine_tune_per_class
                        ]
                        new_data.append(curr_data)
                        new_target.extend([label] * self.fine_tune_per_class)
                    buildin_dataset.data = torch.vstack(new_data)
                    buildin_dataset.targets = torch.tensor(new_target, dtype=torch.long)
                    # import random
                    # random.seed(0)
                    # perm = list(range(per_class * 10))
                    # random.shuffle(perm)
                    # buildin_dataset.data = np.array(new_data)[perm]
                    # buildin_dataset.targets = np.array(new_target)[perm].tolist()
            # """
            _labels = buildin_dataset.targets.clone().detach()

        elif name == "cinic10":
            import glob
            import os
            from shutil import copyfile

            mean = (0.47889522, 0.47227842, 0.43047404)
            std = (0.24205776, 0.23828046, 0.25874835)

            cinic_directory = root
            splitted_path = os.path.normpath(cinic_directory).split(os.sep)
            splitted_path[-1] = "CINIC10-enlarge"
            enlarge_directory = os.path.join(*splitted_path)

            if not os.path.exists(enlarge_directory):
                print("combining trainset and validset...")
                os.makedirs(enlarge_directory)
                os.makedirs(os.path.join(enlarge_directory, "train"))
                os.makedirs(os.path.join(enlarge_directory, "test"))
                # fmt: off
                classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]  # noqa: E501
                # fmt: on
                for c in classes:
                    os.makedirs(os.path.join(enlarge_directory, "train", c))
                    os.makedirs(os.path.join(enlarge_directory, "test", c))

                for s in ("train", "valid", "test"):
                    for c in classes:
                        source_dir = os.path.join(cinic_directory, s, c)
                        filenames = glob.glob("{}/*.png".format(source_dir))
                        for fn in filenames:
                            dest_fn = fn.split("/")[-1]
                            if s == "train" or s == "valid":
                                dest_fn = os.path.join(
                                    enlarge_directory, "train", c, dest_fn
                                )
                            else:
                                dest_fn = os.path.join(
                                    enlarge_directory, "test", c, dest_fn
                                )
                            # if not os.path.islink(dest_fn):
                            #     os.symlink(fn, dest_fn)
                            copyfile(fn, dest_fn)
                print("done.")

            if train:
                transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
                buildin_dataset = datasets.ImageFolder(
                    os.path.join(enlarge_directory, "train"),
                    transform=transform,
                )
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean, std)]
                )
                buildin_dataset = datasets.ImageFolder(
                    os.path.join(enlarge_directory, "test"),
                    transform=transform,
                )
            if self.fine_tune:
                if train:
                    import numpy as np

                    targets = buildin_dataset.targets  # Python List
                    samples = buildin_dataset.samples
                    targets = np.array(targets)
                    samples = np.array(samples)
                    new_samples, new_targets = [], []
                    for label in range(10):
                        curr_data = samples[targets == label][
                            : self.fine_tune_per_class
                        ].tolist()
                        curr_data = [(item[0], item[1]) for item in curr_data]
                        new_samples.extend(curr_data)
                        new_targets.extend([label] * self.fine_tune_per_class)
                    buildin_dataset.samples = new_samples
                    buildin_dataset.targets = new_targets
            _labels = torch.tensor(buildin_dataset.targets, dtype=torch.long)

        elif name == "svhn":
            if train:
                split = "train"
                transform = transforms.Compose(
                    [
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
                split = "test"
            buildin_dataset = datasets.SVHN(
                root=root,
                split=split,
                download=download,
                transform=transform,
            )
            if self.fine_tune:
                if train:
                    import numpy as np

                    targets = buildin_dataset.labels  # numpy array
                    new_data, new_target = [], []
                    for label in range(10):
                        curr_data = buildin_dataset.data[targets == label][
                            : self.fine_tune_per_class
                        ]
                        new_data.append(curr_data)
                        new_target.extend([label] * self.fine_tune_per_class)
                    buildin_dataset.data = np.vstack(new_data)
                    # buildin_dataset.targets = np.array(new_target, dtype=np.int64)
                    buildin_dataset.labels = np.array(new_target, dtype=np.int64)
            _labels = torch.from_numpy(buildin_dataset.labels)

        else:
            raise ValueError("not suported now.")

        return buildin_dataset, _labels


if __name__ == "__main__":
    pass
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
        dataset_name="cifar10",
        root="data",
        train=False,
        download=True,
    )
    _image, _label = cifar_dataset[0]
    print(type(_image), type(_label))
    print(_image.shape)
