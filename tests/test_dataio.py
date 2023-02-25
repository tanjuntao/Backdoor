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

    """
    dummy_dataset = NumpyDataset.dummy_dataset(
        role=Const.ACTIVE_NAME,
        dataset_type=Const.CLASSIFICATION,
        n_samples=100000,
        n_features=10,
        passive_feat_frac=0.5,
    )
    active_trainset, active_testset = NumpyDataset.train_test_split(
        dummy_dataset, test_size=0.2
    )

    dummy_dataset = NumpyDataset.dummy_dataset(
        role=Const.PASSIVE_NAME,
        dataset_type=Const.CLASSIFICATION,
        n_samples=100000,
        n_features=10,
        passive_feat_frac=0.5,
    )
    passive_trainset, passive_testset = NumpyDataset.train_test_split(
        dummy_dataset, test_size=0.2
    )
    for name, dataset in zip(
        ["active_trainset", "active_testset", "passive_trainset", "passive_testset"],
        [active_trainset, active_testset, passive_trainset, passive_testset],
    ):
        print(dataset.ids[-5:])
        # print(dataset.get_dataset()[:3])
        # pd_dataset = pd.DataFrame(dataset.get_dataset(), dtype=np.float16)
        # pd_dataset.to_csv(f"{name}.csv", header=False, index=False)

        np.savetxt(f"{name}.csv", dataset.get_dataset(), delimiter=",")
    print("Done.")
    loaded_dataset = NumpyDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path="./active_trainset.csv",
        dataset_type=Const.CLASSIFICATION
    )
    print(loaded_dataset.ids[:5])
    print(loaded_dataset.ids[-5:])
    """
    dummy_dataset = NumpyDataset.dummy_dataset(
        role=Const.ACTIVE_NAME,
        dataset_type=Const.CLASSIFICATION,
        n_samples=50_0000,
        n_features=100,
        passive_feat_frac=0.5,
    )
    active_trainset, active_testset = NumpyDataset.train_test_split(
        dummy_dataset, test_size=0
    )
    dummy_dataset = NumpyDataset.dummy_dataset(
        role=Const.PASSIVE_NAME,
        dataset_type=Const.CLASSIFICATION,
        n_samples=100_0000,
        n_features=100,
        passive_feat_frac=0.5,
    )
    passive_trainset, passive_testset = NumpyDataset.train_test_split(
        dummy_dataset, test_size=0
    )
    passive_new_dataset = passive_trainset.get_dataset()
    passive_new_dataset[:, 0] = np.arange(20_0000, 120_0000)
    passive_trainset.set_dataset(passive_new_dataset)

    intersection = set(active_trainset.ids) & set(passive_trainset.ids)
    print(f"size of intersection: {len(intersection)}")
    print(active_trainset.get_dataset().shape, passive_trainset.get_dataset().shape)
    np.savetxt("vfl_nn_active.csv", active_trainset.get_dataset(), delimiter=",")
    np.savetxt("vfl_nn_passive.csv", passive_trainset.get_dataset(), delimiter=",")

