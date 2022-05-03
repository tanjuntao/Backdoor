import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

from linkefl.config.vfl_nn_config import NNConfig as Config


class SklearnDataset(Dataset):
    def __init__(self,
                 dataset_name='breast_cancer',
                 role='passive_party',
                 train=True,
                 attacker_features_frac=0.5,
                 permutation=None,
                 transform=None,
                 target_transform=None):

        self.dataset_name = dataset_name
        self.train = train
        self.role = role
        assert self.role in ('passive_party', 'bob'), "role can only take value of 'passive_party or 'bob'"
        self.attacker_features_frac = attacker_features_frac
        self.permutation = permutation
        self.transform = transform
        self.target_transform = target_transform

        if self.dataset_name == 'breast_cancer':
            dataset_obj = load_breast_cancer()
        elif self.dataset_name == 'digits':
            dataset_obj = load_digits()
        else:
            raise ValueError('{} np_dataset can not be supported right now'.format(dataset_name))

        x_train, x_test, y_train, y_test = train_test_split(
            dataset_obj.data,
            dataset_obj.target,
            test_size=0.2,
            random_state=0
        )
        self.x_dataset = x_train if train else x_test
        self.y_dataset = y_train if train else y_test
        self.total_features = x_train.shape[1]

    @property
    def targets(self):
        return torch.from_numpy(self.y_dataset)

    def __len__(self):
        return self.x_dataset.shape[0]

    def __getitem__(self, idx):
        num_alice_features = int(self.attacker_features_frac * self.total_features)
        feature, label = self.x_dataset[idx], self.y_dataset[idx] # type ndarray
        feature = feature.astype(np.float32) # avoid forwarding error
        feature = torch.from_numpy(feature) # type Tensor, shape: [len(feature)]
        feature = feature[self.permutation]

        # if self.transform:
        #     feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)

        if self.role == 'passive_party':
            x_final = feature[:num_alice_features]
            return x_final
        else:
            x_final = feature[num_alice_features:]
            return x_final, label


class TorchDataset(Dataset):
    def __init__(self,
                 dataset_name='fashion_mnist',
                 role='passive_party',
                 train=True,
                 attacker_features_frac=0.5,
                 permutation=None,
                 transform=None,
                 target_transform=None):
        self.dataset_name = dataset_name
        self.role = role
        self.train = train
        self.attacker_features_frac = attacker_features_frac
        self.permutation = permutation
        self.transform = transform
        self.target_transform = target_transform

        if dataset_name == 'fashion_mnist':
            self.torch_dataset = datasets.FashionMNIST(root='raw_data/data',
                                                       train=train,
                                                       download=True,
                                                       transform=transform,
                                                       target_transform=target_transform)
        elif dataset_name == 'mnist':
            self.torch_dataset = datasets.MNIST(root='raw_data/data',
                                                train=train,
                                                download=True,
                                                transform=transform,
                                                target_transform=target_transform)

        elif dataset_name == 'cifar':
            if self.train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
                ])
            self.torch_dataset = datasets.CIFAR10(root='raw_data/data',
                                                  train=train,
                                                  download=True,
                                                  transform=transform,
                                                  target_transform=target_transform)

        elif dataset_name == 'svhn':
            split = 'train' if train else 'validate'
            self.torch_dataset = datasets.SVHN(root='raw_data/data',
                                               split=split,
                                               download=True,
                                               transform=transform,
                                               target_transform=target_transform)

        else:
            pass

    @property
    def targets(self):
        if self.dataset_name == 'svhn':
            return self.torch_dataset.labels
        else:
            return self.torch_dataset.targets # shape: torch.Size([60000])

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, idx):
        image, label = self.torch_dataset[idx] # image shape: torch.Size([1, height, width])
        image_flat = image.view(-1) # shape: torch.Size([height * width])
        image_flat = image_flat[self.permutation] # permute features

        num_alice_features = int(self.attacker_features_frac * len(image_flat))
        if self.role == 'passive_party':
            x_final = image_flat[:num_alice_features]
            return x_final
        else:
            x_final = image_flat[num_alice_features:]
            return x_final, label

        # AVOID USING THIS
        # return x_final if self.role == 'passive_party' else x_final, label


class CSVDataset(Dataset):
    def __init__(self,
                 file_path,
                 role='passive_party',
                 train=True,
                 attacker_features_frac=0.5,
                 permutation=None,
                 transform=None,
                 target_transform=None):

        dataset = np.genfromtxt(file_path, delimiter=',')
        # column1: id, column2: label, column3 and after: features
        self.x_dataset, self.y_dataset = dataset[:, 2:], dataset[:, 1]

        self.role = role
        assert role in ('passive_party', 'bob'), "role can only take value of 'passive_party or 'bob'"
        self.train = train
        self.attacker_features_frac = attacker_features_frac
        self.permutation = permutation
        self.transform = transform
        self.target_transform = target_transform

    @property
    def targets(self):
        return torch.from_numpy(self.y_dataset).int()

    def __len__(self):
        return self.x_dataset.shape[0]

    def __getitem__(self, idx):
        n_samples, n_features = self.x_dataset.shape
        num_alice_features = int(n_features * self.attacker_features_frac)
        feature, label = self.x_dataset[idx], int(self.y_dataset[idx])
        feature = feature.astype(np.float32) # avoid forwarding error
        feature = torch.from_numpy(feature) # tensor shape: Size([len(features)])
        feature = feature[self.permutation]

        # if self.transform:
        #     feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)

        if self.role == 'passive_party':
            return feature[:num_alice_features]
        else:
            return feature[num_alice_features:], label


class CNNFeaturesDataset(Dataset):
    def __init__(self,
                 dataset_name='cifar_vgg13',
                 role='passive_party',
                 train=True,
                 attacker_features_frac=0.5,
                 permutation=None,
                 transform=None,
                 target_transform=None):
        if train:
            _dataset = np.loadtxt('../../data/tabular/cifar_vgg13_train.txt')
        else:
            _dataset = np.loadtxt('../../data/tabular/cifar_vgg13_test.txt')
        self.x_dataset = _dataset[:, :-1]
        self.y_dataset = _dataset[:, -1]

        self.role = role
        self.train = train
        self.attacker_features_frac = attacker_features_frac
        self.permutation = permutation
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.x_dataset.shape[0]

    def __getitem__(self, idx):
        n_samples, n_features = self.x_dataset.shape
        num_alice_features = int(n_features * self.attacker_features_frac)
        feature, label = self.x_dataset[idx], int(self.y_dataset[idx])
        feature = feature.astype(np.float32)
        feature = torch.from_numpy(feature)
        feature = feature[self.permutation]

        if self.target_transform:
            label = self.target_transform(label)
        if self.role == 'passive_party':
            return feature[:num_alice_features]
        else:
            return feature[num_alice_features:], label

class AccumulatedDataset(Dataset):
    def __init__(self, alice_testing_data, labels):
        self.alice_testing_data = alice_testing_data
        self.labels = labels

    def __len__(self):
        return len(self.alice_testing_data)

    def __getitem__(self, idx):
        return self.alice_testing_data[idx], self.labels[idx]


def get_tensor_dataset(dataset_name='fashion_mnist',
                       role='passive_party',
                       train=True,
                       attacker_features_frac=0.5,
                       permutation=None,
                       transform=None,
                       target_transform=None):
    """Load training np_dataset or testing np_dataset.

    Args:
        dataset_name: str. Name of the np_dataset.
        role: str. RSAPSIPassive or bob.
        train: bool. Training or testing.
        attacker_features_frac: float. How much fraction of total features
          does RSAPSIPassive have.
        permutation: torch.randperm. Permutation vector.
        transform: torchvision.transform. How we transfrom features.
        target_transform: torchvision.transorm. How we transform label.

    Returns:
        Dataset for training or testing.

    Raises:
        VauleError: An error will occured if the np_dataset name is not supported.
    """
    print('Loading np_dataset...')
    if dataset_name in ('fashion_mnist', 'mnist', 'cifar', 'svhn'):
        dataset_ = TorchDataset(dataset_name=dataset_name,
                                role=role,
                                train=train,
                                attacker_features_frac=attacker_features_frac,
                                permutation=permutation,
                                transform=transform,
                                target_transform=target_transform)

    elif dataset_name in ('breast_cancer', 'digits'):
        dataset_ =  SklearnDataset(dataset_name=dataset_name,
                                   role=role,
                                   train=train,
                                   attacker_features_frac=attacker_features_frac,
                                   transform=transform,
                                   target_transform=target_transform)

    elif dataset_name in ('give_me_some_credit', 'census_income'):
        if dataset_name == 'give_me_some_credit' and train:
            file_path = '../../data/tabular/give_me_some_credit_train.csv'
        elif dataset_name == 'give_me_some_credit' and not train:
            file_path = '../../data/tabular/give_me_some_credit_test.csv'
        elif dataset_name == 'census_income' and train:
            file_path =  '../../data/tabular/census_income_train.csv'
        else:
            file_path = '../../data/tabular/census_income_test.csv'

        dataset_ =  CSVDataset(file_path=file_path,
                               role=role,
                               train=train,
                               attacker_features_frac=attacker_features_frac,
                               permutation=permutation,
                               transform=transform,
                               target_transform=target_transform)

    elif dataset_name == 'cifar_vgg13':
        dataset_ =  CNNFeaturesDataset(dataset_name=dataset_name,
                                       role=role,
                                       train=train,
                                       attacker_features_frac=attacker_features_frac,
                                       permutation=permutation,
                                       transform=transform,
                                       target_transform=target_transform)

    else:
        raise ValueError('Unsupported np_dataset: {}'.format(dataset_name))

    print('Done!')
    return dataset_


if __name__ == '__main__':
    pass

    # TODO: import the config object here
    # np_dataset = get_dataset(dataset_name='mnist',
    #                       permutation=Config.PERMUTATION,
    #                       transform=ToTensor())
    # print(np_dataset[0])
