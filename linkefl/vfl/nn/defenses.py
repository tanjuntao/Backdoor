import numpy as np
import torch


class TensorPruner:
    def __init__(self, zip_percent):
        self.thresh_hold = 0.0
        self.zip_percent = zip_percent

    def update_thresh_hold(self, tensor):
        tensor_copy = tensor.clone().detach()
        tensor_copy = torch.abs(tensor_copy)
        survivial_values = torch.topk(
            tensor_copy.reshape(1, -1),
            int(tensor_copy.reshape(1, -1).shape[1] * self.zip_percent),
        )
        self.thresh_hold = survivial_values[0][0][-1]

    def prune_tensor(self, tensor):
        background_tensor = torch.zeros(tensor.shape).to(tensor.device)
        tensor = torch.where(abs(tensor) > self.thresh_hold, tensor, background_tensor)
        return tensor


# Multistep gradient
class DiscreteGradient:
    def __init__(self, bins_num, bound_abs=3e-2):
        self.bins_num = bins_num
        self.bound_abs = bound_abs

    def apply(self, tensor):
        max_min = 2 * self.bound_abs
        interval = max_min / self.bins_num
        tensor_ratio_interval = torch.div(tensor, interval)
        tensor_ratio_interval_rounded = torch.round(tensor_ratio_interval)
        tensor_multistep = tensor_ratio_interval_rounded * interval
        return tensor_multistep


class DistanceCorrelationLoss(torch.nn.modules.loss._Loss):
    # def __init__(self, dcor_weight):
    #     self.dcor_weight = dcor_weight

    def forward(self, input_data, intermediate_data):
        input_data = input_data.view(input_data.size(0), -1)
        intermediate_data = intermediate_data.view(intermediate_data.size(0), -1)

        # Get A matrices of data
        A_input = self._A_matrix(input_data)
        A_intermediate = self._A_matrix(intermediate_data)

        # Get distance variances
        input_dvar = self._distance_variance(A_input)
        intermediate_dvar = self._distance_variance(A_intermediate)

        # Get distance covariance
        dcov = self._distance_covariance(A_input, A_intermediate)

        # Put it together
        dcorr = dcov / (input_dvar * intermediate_dvar).sqrt()

        return dcorr
        # return dcorr * self.dcor_weight

    def _distance_covariance(self, a_matrix, b_matrix):
        return (a_matrix * b_matrix).sum().sqrt() / a_matrix.size(0)

    def _distance_variance(self, a_matrix):
        return (a_matrix**2).sum().sqrt() / a_matrix.size(0)

    def _A_matrix(self, data):
        distance_matrix = self._distance_matrix(data)

        row_mean = distance_matrix.mean(dim=0, keepdim=True)
        col_mean = distance_matrix.mean(dim=1, keepdim=True)
        data_mean = distance_matrix.mean()

        return distance_matrix - row_mean - col_mean + data_mean

    def _distance_matrix(self, data):
        n = data.size(0)
        distance_matrix = torch.zeros((n, n))
        flag = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if flag[j][i] != 0:  # already computed
                    flag[i][j] = flag[j][i].clone()
                    continue
                row_diff = data[i] - data[j]
                distance_matrix[i, j] = (row_diff**2).sum()
                flag[i][j] = 1

        return distance_matrix


class LabelDP:
    """
    Label differential privacy module.

    This class uses the Random Response algorithm to create
    differentially private label based on the input label.
    Currently only support binary labels (dim = 1 or 2) and onehot labels (dim = 2).

    Args:
        eps (Union[int, float]): the privacy parameter, representing the
        level of differential privacy protection.

    Inputs:
        - **label** (Tensor) - a batch of labels to be made differentially private.

    Outputs:
        Tensor, has the same shape and data type as `label`.

    Raises:
        TypeError: If `eps` is not a float or int.
        TypeError: If `label` is not a Tensor.
        ValueError: If `eps` is less than zero.
    """

    def __init__(self, eps) -> None:
        self.eps = eps

    def __call__(self, label: torch.Tensor):
        """
        input a label batch, output a perturbed label batch satisfying
        label differential privacy.
        """
        if not isinstance(label, torch.Tensor):
            raise TypeError(f"The label must be a Tensor, but got {type(label)}")

        ones_cnt = np.sum(label.numpy() == 1)
        zeros_cnt = np.sum(label.numpy() == 0)
        if ones_cnt + zeros_cnt != label.numel():
            raise ValueError(
                "Invalid label form: the elements should be either 0 or 1."
            )

        if label.ndim == 1 or (label.ndim == 2 and label.shape[1] == 1):
            flip_prob = 1 / (np.exp(self.eps) + 1)
            binomial = np.random.binomial(1, flip_prob, label.shape)
            dp_label = (label - torch.tensor(binomial, dtype=label.dtype)).abs()
        elif label.ndim == 2:
            if ones_cnt != len(label):
                raise ValueError(
                    "Invalid one-hot form: each label should contain only a single 1."
                )
            keep_prob = np.exp(self.eps) / (np.exp(self.eps) + label.shape[1] - 1)
            flip_prob = 1 / (np.exp(self.eps) + label.shape[1] - 1)
            prob_array = (
                label * (keep_prob - flip_prob)
                + torch.tensor(np.ones(label.shape)) * flip_prob
            )
            dp_index = np.array(
                [
                    np.random.choice(label.shape[1], p=prob / sum(prob))
                    for prob in prob_array
                ]
            )
            dp_label = torch.tensor(np.eye(label.shape[1])[dp_index], dtype=label.dtype)
        else:
            raise ValueError("Invalid label dim: the dim must be 1 or 2.")

        return dp_label
