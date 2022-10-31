from abc import ABC, abstractmethod


class TransformComponent(ABC):
    @abstractmethod
    def fit(self, dataset, role):
        pass


class ModelComponent(ABC):
    @abstractmethod
    def fit(self, trainset, validset, role):
        pass

    @abstractmethod
    def score(self, testset, role):
        pass
