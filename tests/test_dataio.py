from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale


if __name__ == "__main__":
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