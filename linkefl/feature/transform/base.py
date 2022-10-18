from abc import ABC, abstractmethod


class BaseTransform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def fit(self, dataset, role):
        return self(dataset, role)
