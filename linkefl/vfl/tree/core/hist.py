import numpy as np

from linkefl.base import BaseCryptoSystem
from linkefl.common.const import Const


class ActiveHist:
    def __init__(self, task, n_labels, bin_gh):
        self.task = task
        self.n_labels = n_labels
        self.bin_gh = bin_gh

    def __sub__(self, other):
        """overload __sub__"""

        return ActiveHist(
            task=self.task, n_labels=self.n_labels, bin_gh=self.bin_gh - other.bin_gh
        )

    @classmethod
    def compute_hist(cls, task, n_labels, sample_tag, bin_index, gh):
        """compute hist with data on active party"""

        try:
            bin_num = bin_index.max() + 1
        except Exception:
            # need to be more elegant
            bin_num = 16

        if task == "binary" or task == "regression":
            bin_gh = np.zeros((bin_index.shape[1], bin_num, 2), dtype=gh.dtype)
        elif task == "multi":
            bin_gh = np.zeros(
                (bin_index.shape[1], bin_num, n_labels, 2), dtype=gh.dtype
            )
        else:
            raise ValueError("No such task label.")

        for sample_i, has_sample in enumerate(sample_tag):
            if not has_sample:
                continue
            for feature_i, bin_i in enumerate(bin_index[sample_i]):
                bin_gh[feature_i][bin_i] += gh[sample_i]

        return cls(task, n_labels, bin_gh)

    @classmethod
    def decrypt_hist(
        cls,
        task,
        n_labels,
        bin_gh_enc,
        h_length,
        r,
        crypto_system: BaseCryptoSystem,
        pool,
    ):
        """decrypt hist received from passive party, binary only"""

        bin_gh_int = crypto_system.decrypt_data(bin_gh_enc, pool=pool)

        return cls.splitgh_hist(task, n_labels, bin_gh_int, h_length, r)

    @classmethod
    def decompress_hist(
        cls,
        task,
        n_labels,
        capacity,
        bin_gh_compress,
        h_length,
        gh_length,
        r,
        crypto_system,
        pool,
    ):
        """decompress and decrypt hist received from passive party, binary only"""

        shape = bin_gh_compress["shape"]
        target = bin_gh_compress["target"]
        bin_nonzero_compress = bin_gh_compress["data"]

        bin_nonzero_compress = crypto_system.decrypt_data(
            bin_nonzero_compress, pool=pool
        )
        bin_nonzero = np.empty((capacity, len(bin_nonzero_compress)), dtype=object)

        for i in range(capacity):
            bin_nonzero[i] = bin_nonzero_compress % (1 << gh_length)
            bin_nonzero_compress >>= gh_length

        bin_nonzero[bin_nonzero > (1 << (gh_length - 1))] -= 1 << gh_length
        bin_nonzero = bin_nonzero.flatten()

        bin_gh_int = np.zeros(shape, dtype=object)
        bin_gh_int[target] = bin_nonzero[: target[0].shape[0]]

        return cls.splitgh_hist(task, n_labels, bin_gh_int, h_length, r)

    @classmethod
    def decompress_multi_hist(
        cls,
        task,
        n_labels,
        capacity,
        bin_gh_compress_multi,
        h_length,
        gh_length,
        r,
        crypto_type,
        crypto_system,
        pool,
    ):
        """decrypt and decompress hist received from passive party, multi only"""
        if crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
            bin_gh_compress_multi = crypto_system.decrypt_data(
                bin_gh_compress_multi, pool=pool
            )

        bin_gh_int = np.zeros(
            (bin_gh_compress_multi.shape[0], bin_gh_compress_multi.shape[1], n_labels),
            dtype=object,
        )

        last_count = n_labels % capacity

        for i in range(bin_gh_compress_multi.shape[0]):
            for j in range(bin_gh_compress_multi.shape[1]):
                bin_gh_label_compress = bin_gh_compress_multi[i][j]
                target = []
                for k, big_num in enumerate(bin_gh_label_compress):
                    if k == len(bin_gh_label_compress) - 1:
                        gh_num = last_count
                    else:
                        gh_num = capacity
                    temp = []
                    for _ in range(gh_num):
                        gh = big_num % (1 << gh_length)
                        if gh > (1 << (gh_length - 1)):
                            gh -= 1 << gh_length
                        big_num = big_num >> gh_length
                        temp.append(gh)
                    temp.reverse()
                    target += temp
                bin_gh_int[i, j, :] = target

        bin_gradient = np.array(bin_gh_int >> h_length, dtype=np.float64) / (1 << r)
        bin_hessian = np.array(bin_gh_int % (1 << h_length), dtype=np.float64) / (
            1 << r
        )

        bin_gh = np.stack((bin_gradient, bin_hessian), axis=3)

        return cls(task, n_labels, bin_gh)

    @classmethod
    def splitgh_hist(cls, task, n_labels, bin_gh_int, h_length, r):
        """split unencrypted gh and transform g, h from big int back to float"""

        bin_gradient = np.array(bin_gh_int >> h_length, dtype=np.float64) / (1 << r)
        bin_hessian = np.array(bin_gh_int % (1 << h_length), dtype=np.float64) / (
            1 << r
        )

        bin_gh = np.stack((bin_gradient, bin_hessian), axis=2)

        return cls(task, n_labels, bin_gh)


class PassiveHist:
    def __init__(self, task, sample_tag, bin_index, gh_data):
        """Passive Party Hist class, construct and compress hist data

        Args:
            task: binary or multi
            sample_tag: denote whether sample shows in current hist, size = sample
            bin_index: bin index of each feature point in the
                complete feature hist (a column), size = sample * feature
            gh_data: gh_int received from active party, size = sample
        """

        self.task = task
        self.bin_gh_data = self._set_hist(sample_tag, bin_index, gh_data)

    def _set_hist(self, sample_tag, bin_index, gh_data):
        """compute hist"""

        bin_num = bin_index.max() + 1  # max bin number in all hist

        if self.task == "binary" or self.task == "regression":
            bin_gh_data = np.zeros((bin_index.shape[1], bin_num), dtype=gh_data.dtype)
        elif self.task == "multi":
            bin_gh_data = np.zeros(
                (bin_index.shape[1], bin_num, gh_data.shape[1]), dtype=gh_data.dtype
            )
        else:
            raise ValueError("Not support task label.")

        for sample_i, has_sample in enumerate(sample_tag):
            if not has_sample:
                continue
            for feature_i, bin_i in enumerate(bin_index[sample_i]):
                bin_gh_data[feature_i][bin_i] += gh_data[sample_i]

        return bin_gh_data

    def compress(self, capacity, padding):
        """compress hist to send to active party"""

        target = np.nonzero(self.bin_gh_data)  # non-zero data positions

        bin_nonzero = self.bin_gh_data[target]  # extract all non-zero data

        # pad 0 at end so that len(bin_nonzero) % capacity = 0,
        # and then reshape to (capacity, -1)
        bin_nonzero = np.pad(
            bin_nonzero, (0, (-len(bin_nonzero)) % capacity), "constant"
        ).reshape((capacity, -1))

        for i in range(capacity):
            bin_nonzero[i] *= 1 << (padding * i)

        bin_gh_compress = {
            "shape": self.bin_gh_data.shape,
            "target": target,
            "data": np.sum(bin_nonzero, axis=0),
        }

        return bin_gh_compress
