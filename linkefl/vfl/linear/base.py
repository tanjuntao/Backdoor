from abc import ABC, abstractmethod

import numpy as np


class BaseLinear(ABC):
    def __init__(self, learning_rate, random_state):
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _init_weights(self, size):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        else:
            np.random.seed(None)
        params = np.random.normal(0, 1.0, size)
        return params

    def _gradient_descent(self, params, grad):
        params -= self.learning_rate * grad

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError('should not call abstract class method')

    @abstractmethod
    def _sync_pubkey(self):
        pass

    @abstractmethod
    def _grad(self, residue, batch_idxes):
        pass

    @abstractmethod
    def train(self, trainset, testset):
        pass

    @abstractmethod
    def validate(self, valset):
        pass

    @abstractmethod
    def predict(self, testset):
        pass






