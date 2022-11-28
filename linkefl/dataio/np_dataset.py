import numpy as np

from linkefl.base import BaseTransformComponent
from linkefl.dataio.common_dataset import CommonDataset


class NumpyDataset(CommonDataset):
    def __init__(self,
                 role: str,
                 raw_dataset: np.ndarray,
                 dataset_type: str,
                 transform: BaseTransformComponent = None,
    ):
        super(NumpyDataset, self).__init__(
            role=role,
            raw_dataset=raw_dataset,
            dataset_type=dataset_type,
            transform=transform
        )
