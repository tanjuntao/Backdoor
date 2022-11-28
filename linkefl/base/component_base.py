from abc import ABC, abstractmethod


class BasePSIComponent(ABC):
    @abstractmethod
    def fit(self, dataset, role):
        pass

    @abstractmethod
    def run(self, ids):
        pass


class BaseTransformComponent(ABC):
    @abstractmethod
    def __call__(self, dataset, role):
        pass

    def fit(self, dataset, role):
        return self.__call__(dataset, role)


class BaseModelComponent(ABC):
    @abstractmethod
    def fit(self, trainset, validset, role):
        pass

    @abstractmethod
    def score(self, testset, role):
        pass
