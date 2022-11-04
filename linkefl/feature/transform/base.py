from abc import ABC, abstractmethod

from linkefl.pipeline.base import TransformComponent


class BaseTransform(TransformComponent, ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def fit(self, dataset, role):
        return self(dataset, role)
