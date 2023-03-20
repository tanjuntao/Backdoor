from typing import List, Optional

import numpy as np

from linkefl.base import BaseTransformComponent
from linkefl.dataio.common_dataset import CommonDataset


class NumpyDataset(CommonDataset):
    def __init__(
        self,
        *,
        role: str,
        raw_dataset: np.ndarray,
        header: List[str],
        dataset_type: str,
        transform: Optional[BaseTransformComponent] = None,
        header_type: Optional[List[str]] = None,
    ):
        super(NumpyDataset, self).__init__(
            role=role,
            raw_dataset=raw_dataset,
            header=header,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )
