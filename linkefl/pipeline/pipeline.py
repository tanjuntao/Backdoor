from typing import Union

from linkefl.base import BasePSIComponent, BaseTransformComponent, BaseModelComponent
from linkefl.dataio import NumpyDataset, TorchDataset


class PipeLine:
    def __init__(self, components, role):
        self._check_params(components)

        self.components = components
        self.role = role

    @staticmethod
    def _check_params(components):
        assert isinstance(components[0], BasePSIComponent)
        for component in components[1:-1]:
            assert isinstance(component, BaseTransformComponent)
        assert isinstance(components[-1], BaseModelComponent)

    def fit(self, trainset: Union[NumpyDataset, TorchDataset], validset: Union[NumpyDataset, TorchDataset]):
        for component in self.components[:-1]:
            trainset = component.fit(trainset, role=self.role)
            validset = component.fit(validset, role=self.role)
        self.components[-1].fit(trainset, validset, role=self.role)

    def score(self, testset: Union[NumpyDataset, TorchDataset]):
        for component in self.components[:-1]:
            testset = component.fit(testset, role=self.role)
        self.components[-1].score(testset, role=self.role)
