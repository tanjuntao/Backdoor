from abc import abstractmethod, ABC


class BaseDataset(ABC):
    @property
    @abstractmethod
    def ids(self):
        pass

    @property
    @abstractmethod
    def features(self):
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

    @property
    @abstractmethod
    def n_samples(self):
        pass

    @property
    @abstractmethod
    def n_features(self):
        pass

    @abstractmethod
    def describe(self):
        pass

