import numpy as np
from abc import ABC, abstractmethod


FLOAT_ZERO = 1e-8


# Base loss class
class Loss(ABC):
    """Abstract Loss Class"""

    def __init__(self):
        pass

    @abstractmethod
    def loss(self, label, y_pred):
        pass

    @abstractmethod
    def gradient(self, label, y_pred):
        pass

    @abstractmethod
    def hessian(self, label, y_pred):
        pass


# Classification loss function
class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y, y_prob):
        """
        The cross-entropy loss class for binary classification
            Formula : -(sum(y * log(y_prob) + (1 - y) * log(1 - y_prob)) / N)

        Args:
            y: the input data's labels
            y_prob: the predict probability.

        Returns:
            log_loss : float, the binary cross entropy loss
        """
        loss = - y * np.log(y_prob) - (1 - y) * np.log(1 - y_prob)

        return loss

    def gradient(self, y, y_prob):
        """
        Compute the grad of sigmoid cross entropy function
            Formula : gradient = y_pred - y

        Args:
            y: the input data's labels
            y_prob: the predict probability.

        Returns:
            gradient : float, the gradient of binary cross entropy loss
        """
        return np.array(y - y_prob)

    def hessian(self, y, y_prob):
        """"
        Compute the hessian(second order derivative of sigmoid cross entropy loss
            Formula : hessian = y_prob * (1 - y_prob)

        Parameters
        ----------
        y : int, just use for function interface alignment
        y_prob : float, the predict probability

        Returns
        -------
        hess : float, the hessian of binary cross entropy loss
        """
        return np.array(y_prob * (1 - y_prob))


class MultiCrossEntropyLoss(Loss):
    """The loss function of multi-classification.

    Parameters:
        label: np.array, dataNum * classNum, each row is a one-hot vector.
        y_pred: np.array, dataNum * class_num, each row is a vector after softmax.
    """

    def __init__(self):
        super().__init__()

    def loss(self, label, y_pred):
        """Calculate multi-class cross entropy loss."""
        loss = []

        for i in range(label.shape[0]):
            loss_sample = np.sum(-1 * label[i, :] * np.log(y_pred[i, :]))
            loss.append(loss_sample)

        return np.array(loss)

    def gradient(self, label, y_pred):
        """Calculate gradient of MCE loss"""
        grad = []

        for i in range(label.shape[0]):
            grad_sample = y_pred[i, :] - label[i, :]
            grad.append(grad_sample)

        return np.array(grad)

    def hessian(self, label, y_pred):
        """Calculate hessian of MCE loss"""

        hess = []

        for i in range(label.shape[0]):
            hess_sample = y_pred[i, :] * (1 - y_pred[i, :])
            hess.append(hess_sample)

        return np.array(hess)


class BalancedCE(object):
    """CrossEntropyLoss with weighting factor α;

    Args:
        α ∈ [0, 1] for class 1 and 1 − α for class −1.
    """

    def __init__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("The value of alpha is out of range.")

        self.alpha = alpha

    def loss(self, label, sigmoid_pred):
        """Calculate Balanced CE loss"""

        loss = -self.alpha * label * np.log(sigmoid_pred) - (1 - self.alpha) * (1 - sigmoid_pred) * np.log(
            (1 - sigmoid_pred)
        )

        return loss

    def grad(self, label, sigmoid_pred):
        """Calculate gradient of Balanced CE loss"""
        grad = -(self.alpha * label * (1 - sigmoid_pred)) + (1 - self.alpha) * (1 - label) * sigmoid_pred

        return np.array(grad)

    def hessian(self, label, sigmoid_pred):
        """Calculate hessian of Balanced CE loss"""
        hessian = (1 - self.alpha - label + 2 * self.alpha * label) * sigmoid_pred * (1 - sigmoid_pred)
        # Avoid having all the second derivatives equal to 0
        hessian[hessian == 0] = float(1e-16)

        return np.array(hessian)


class FocalLoss(object):
    """Weight balance focal loss:"""

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def loss(self, label, sigmoid_pred):
        """Calculate weight balance Focal loss"""

        loss = self.alpha * (-label * np.log(sigmoid_pred) * ((1 - sigmoid_pred) ** self.gamma)) - (1 - self.alpha) * (
                1 - label
        ) * np.log(1 - sigmoid_pred) * (sigmoid_pred ** self.gamma)

        return loss

    def grad(self, label, sigmoid_pred):
        """Calculate gradient of weight balance focal loss"""

        alpha, gamma = self.alpha, self.gamma
        y, p = label, sigmoid_pred

        grad = (
                p
                * (1 - p)
                * (
                        alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p)
                        - alpha * y * (1 - p) ** gamma / p
                        - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p
                        + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)
                )
        )

        return np.array(grad)

    def hessian(self, label, sigmoid_pred):
        """Calculate hessian of weight balance focal loss"""

        alpha, gamma = self.alpha, self.gamma
        y, p = label, sigmoid_pred

        hess = (
                p
                * (1 - p)
                * (
                        p
                        * (1 - p)
                        * (
                                -alpha * gamma ** 2 * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2
                                + alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2
                                + 2 * alpha * gamma * y * (1 - p) ** gamma / (p * (1 - p))
                                + alpha * y * (1 - p) ** gamma / p ** 2
                                - gamma ** 2 * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2
                                + 2 * gamma * p ** gamma * (1 - alpha) * (1 - y) / (p * (1 - p))
                                + gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2
                                + p ** gamma * (1 - alpha) * (1 - y) / (1 - p) ** 2
                        )
                        - p
                        * (
                                alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p)
                                - alpha * y * (1 - p) ** gamma / p
                                - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p
                                + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)
                        )
                        + (1 - p)
                        * (
                                alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p)
                                - alpha * y * (1 - p) ** gamma / p
                                - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p
                                + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)
                        )
                )
        )

        return np.array(hess)


# Regression loss function
class MeanSquaredErrorLoss(Loss):
    """The loss function of the regression task.

    Parameters:
        label: np.array, dataNum * classNum, each row is a one-hot vector.
        y_pred: np.array, dataNum * class_num, each row is a vector after softmax.

    Returns:
        loss: a list consist of float value.
    """

    def __init__(self):
        super().__init__()

    def loss(self, y, y_pred):
        """Calculate MSE loss"""
        loss = (y - y_pred) * (y - y_pred)
        return loss

    def gradient(self, y, y_pred):
        """Calculate gradient of MSE loss"""
        return np.array(2 * (y_pred-y))

    def hessian(self, y, y_pred):
        """Calculate hessian of MSE loss"""
        if isinstance(y, np.ndarray) or isinstance(y_pred, np.ndarray):
            shape = (y - y_pred).shape
            return np.full(shape, 2)
        else:
            return 2


class LeastAbsoluteErrorLoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y, y_pred):
        loss = np.abs(y-y_pred)
        return loss

    def gradient(self, y, y_pred):
        if isinstance(y, np.ndarray) or isinstance(y_pred, np.ndarray):
            diff = y_pred - y
            diff[diff > FLOAT_ZERO] = 1
            diff[diff < FLOAT_ZERO] = -1
            diff[np.abs(diff) <= FLOAT_ZERO] = 0

            return diff
        else:
            diff = y_pred - y
            if diff > FLOAT_ZERO:
                return 1
            elif diff < FLOAT_ZERO:
                return -1
            else:
                return 0

    def hessian(self, y, y_pred):
        if isinstance(y, np.ndarray) or isinstance(y_pred, np.ndarray):
            shape = (y-y_pred).shape
            return np.full(shape, 1)
        else:
            return 1


class HubelLoss(Loss):
    def __init__(self, delta=None):
        super().__init__()
        if delta is None:
            self.delta = FLOAT_ZERO
        else:
            self.delta = delta

        if np.abs(self.delta) < FLOAT_ZERO:
            self.delta = FLOAT_ZERO

    def loss(self, y, y_pred):
        loss = self.delta ** 2 * (np.sqrt(1 + ((y_pred - y) / self.delta) ** 2) - 1)
        return loss

    def gradient(self, y, y_pred):
        diff = y_pred - y
        return diff / np.sqrt(1.0 + diff * diff / (self.delta ** 2))

    def hessian(self, y, y_pred):
        diff = y_pred - y
        return 1.0 / (1.0 + diff * diff / (self.delta ** 2)) ** 1.5


class FairLoss(Loss):
    def __init__(self, c=None):
        super().__init__()
        if c is None:
            self.c = FLOAT_ZERO
        else:
            self.c = c

        if np.abs(self.c) < FLOAT_ZERO:
            self.c = FLOAT_ZERO

    def loss(self, y, y_pred):
        loss = self.c * np.abs(y_pred - y) - self.c ** 2 * np.log(np.abs(y_pred - y) / self.c + 1)
        return loss

    def gradient(self, y, y_pred):
        diff = y_pred - y
        return self.c * diff / (np.abs(diff) + self.c)

    def hessian(self, y, y_pred):
        diff = y_pred - y
        return self.c ** 2 / (np.abs(diff) + self.c) ** 2


class LogCoshLoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y, y_pred):
        loss = np.log(np.cosh(y_pred - y))
        return loss

    def gradient(self, y, y_pred):
        return np.tanh(y_pred - y)

    def hessian(self, y, y_pred):
        return 1 - np.tanh(y_pred - y) ** 2


class TweedieLoss(Loss):
    def __init__(self, rho=None):
        super().__init__()

        if rho is None:
            self.rho = FLOAT_ZERO
        else:
            self.rho = rho

        if np.abs(self.rho) < FLOAT_ZERO:
            self.rho = FLOAT_ZERO

    def loss(self, y, y_pred):
        loss = - y * np.exp(1 - self.rho) * np.log(max(y_pred, FLOAT_ZERO)) / (1 - self.rho) + \
               np.exp(2 - self.rho) * np.log(max(FLOAT_ZERO, y_pred)) / (2 - self.rho)
        return loss

    def gradient(self, y, y_pred):
        return -y * np.exp(1 - self.rho) * y_pred + np.exp(2 - self.rho) * y_pred

    def hessian(self, y, y_pred):
        return -y * (1 - self.rho) * np.exp(1 - self.rho) * y_pred + (2 - self.rho) * np.exp(2 - self.rho) * y_pred