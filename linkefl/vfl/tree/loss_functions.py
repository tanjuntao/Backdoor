from abc import ABC, abstractmethod

import numpy as np


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


class MSELoss(Loss):
    """The loss function of the regression task.

    Parameters:
        label: np.array, dataNum * classNum, each row is a one-hot vector.
        y_pred: np.array, dataNum * class_num, each row is a vector after softmax.

    Returns:
        loss: a list consist of float value.
    """

    def __init__(self):
        super().__init__()

    def loss(self, label, y_pred):
        """Calculate MSE loss"""

        loss = 0.5 * np.power((label - y_pred), 2)
        return loss

    def gradient(self, label, y_pred):
        """Calculate gradient of MSE loss"""

        grad = -(label - y_pred)
        return np.array(grad)

    def hessian(self, label, y_pred):
        """Calculate hessian of MSE loss"""

        hess = np.ones_like(label)
        return hess


class CrossEntropyLoss(Loss):
    """The loss function of Binary classification.

    Parameters:
        label: np.array, dataNum * 1.
        y_pred: np.array, dataNum * 1, after sigmoid.
    """

    def __init__(self):
        super().__init__()

    def loss(self, label, y_pred):
        """Calculate binary cross entropy loss"""

        loss = -label * np.log(y_pred) - (1 - label) * np.log(1 - y_pred)
        # loss = label * np.log(1 + np.exp(-sigmoid_pred)) + \
        #         (1 - label) * np.log(1 + np.exp(sigmoid_pred))

        return loss

    def gradient(self, label, y_pred):
        """Calculate gradient of CE loss"""

        grad = np.array(y_pred - label)

        return grad

    def hessian(self, label, y_pred):
        """Calculate hessian of CE loss"""

        hess = np.array(y_pred * (1 - y_pred))
        # Avoid having all the second derivatives equal to 0；但是sigmoid的值不会等于0或者1，这里真的有必要吗
        # TODO: 做测试
        if hess.sum() == 0:
            print("All value of hessian is zero")
            temp = y_pred * (1 - y_pred)
            print(f"y_pred * (1 - y_pred) is : \n{temp}")
            print(f"hessian is : \n{hess}")

        hess[hess == 0] = float(1e-16)

        return hess


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
        ) * np.log(1 - sigmoid_pred) * (sigmoid_pred**self.gamma)

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
                - gamma * p**gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p
                + p**gamma * (1 - alpha) * (1 - y) / (1 - p)
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
                    -alpha * gamma**2 * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2
                    + alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2
                    + 2 * alpha * gamma * y * (1 - p) ** gamma / (p * (1 - p))
                    + alpha * y * (1 - p) ** gamma / p**2
                    - gamma**2 * p**gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p**2
                    + 2 * gamma * p**gamma * (1 - alpha) * (1 - y) / (p * (1 - p))
                    + gamma * p**gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p**2
                    + p**gamma * (1 - alpha) * (1 - y) / (1 - p) ** 2
                )
                - p
                * (
                    alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p)
                    - alpha * y * (1 - p) ** gamma / p
                    - gamma * p**gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p
                    + p**gamma * (1 - alpha) * (1 - y) / (1 - p)
                )
                + (1 - p)
                * (
                    alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p)
                    - alpha * y * (1 - p) ** gamma / p
                    - gamma * p**gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p
                    + p**gamma * (1 - alpha) * (1 - y) / (1 - p)
                )
            )
        )

        return np.array(hess)
