import numpy as np
import pandas as pd

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset, TorchDataset
from linkefl.feature.transform import scale


if __name__ == "__main__":
    """
    abs_path = "/Users/tanjuntao/Desktop/cancer.csv"
    np_dataset = NumpyDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path=abs_path,
        dataset_type=Const.CLASSIFICATION,
    )
    np_dataset = scale(np_dataset)
    print(isinstance(np_dataset, NumpyDataset))
    print(np_dataset.n_samples, np_dataset.n_features)
    print(np_dataset.header)
    print(np_dataset.header_type)
    print(np_dataset.describe())

    print(np_dataset.get_dataset().shape)
    print(np_dataset.mappings)
    pd_dataset = pd.DataFrame(np_dataset.get_dataset(), dtype=np.float16)
    pd_dataset.to_csv("cancer.csv", header=False, index=False)
    """

    # Export vfl_nn dataset to csv file
    """
    dataset_name = "tab_fashion_mnist"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_trainset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.ACTIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_trainset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.PASSIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_trainset.set_dataset(active_trainset.get_dataset()[:12000])
    passive_trainset.set_dataset(passive_trainset.get_dataset()[:12000])
    active_trainset = pd.DataFrame(data=active_trainset.get_dataset().numpy(), dtype=np.float16)
    passive_trainset = pd.DataFrame(data=passive_trainset.get_dataset().numpy(), dtype=np.float16)
    active_trainset.to_csv("vfl_nn_active.csv", index=False, header=False)
    passive_trainset.to_csv("vfl_nn_passive.csv", index=False, header=False)
    """

    # Export stability-validation dataset (census)
    """
    _dataset_name = "census"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_trainset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_trainset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_trainset = pd.DataFrame(data=active_trainset.get_dataset(), dtype=np.float16)
    passive_trainset = pd.DataFrame(data=passive_trainset.get_dataset(), dtype=np.float16)
    active_trainset.to_csv("vfl_logreg_active.csv", index=False, header=False)
    passive_trainset.to_csv("vfl_logreg_passive.csv", index=False, header=False)
    """
    pass

