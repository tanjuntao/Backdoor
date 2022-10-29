from __future__ import annotations  # python >= 3.7, give type hint before definition

import numpy as np

from linkefl.dataio.common_dataset import CommonDataset
# avoid circular importing
# from linkefl.feature.transform import BaseTransform


class NumpyDataset(CommonDataset):
    def __init__(self,
                 role: str,
                 raw_dataset: np.ndarray,
                 dataset_type: str,
                 # transform: BaseTransform = None
                 transform=None,
    ):
        super(NumpyDataset, self).__init__(
            role=role,
            raw_dataset=raw_dataset,
            dataset_type=dataset_type,
            transform=transform
        )
