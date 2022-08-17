from multiprocessing import Pool

import numpy as np

from linkefl.common.const import Const
from linkefl.crypto.base import CryptoSystem
from linkefl.messenger.base import Messenger
from linkefl.vfl.tree.hist import ActiveHist
from linkefl.vfl.tree.data_functions import wrap_message
from linkefl.vfl.tree.train_functions import gh_packing, find_split, leaf_weight, gh_compress_multi, leaf_weight_multi


class _DecisionNode:
    def __init__(self, party_id=None, record_id=None, left_branch=None, right_branch=None, value=None):
        # 查询时使用
        self.party_id = party_id
        self.record_id = record_id

        # 创建中间节点时使用
        self.left_branch = left_branch
        self.right_branch = right_branch

        # 叶节点信息
        self.value = value


class DecisionTree:
    def __init__(
        self,
        task: str,
        n_labels: int,
        crypto_type: str,
        crypto_system: CryptoSystem,
        messenger: Messenger,
        *,
        compress: bool = False,
        max_depth: int = 4,
        reg_lambda: float = 0.1,
        min_split_samples: int = 3,
        min_split_gain: float = 1e-7,
        fix_point_precision: int = 53,
        sampling_method: str = "uniform",
        pool: Pool = None,
    ):
        """Decision Tree class to create a single tree

        Args:
            task: binary or multi
            n_labels: number of labels, should be 2 if task is set as binary
            compress: can only be enabled when task is set as binary
            max_depth: max depth of a tree, including root
            reg_lambda: used to compute gain and leaf weight
            min_split_samples: minimum samples required to split
            min_split_gain: minimum gain required to split
            fix_point_precision: bit length to preserve when converting float to int
            sampling_method: uniform
            pool: not supported yet
        """

        self.task = task
        self.n_labels = n_labels
        self.crypto_type = crypto_type
        self.crypto_system = crypto_system
        self.messenger = messenger

        self.compress = compress
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.min_split_samples = min_split_samples
        self.min_split_gain = min_split_gain
        self.fix_point_precision = fix_point_precision
        self.sampling_method = sampling_method
        self.pool = pool

        # given when training starts
        self.bin_index = None
        self.bin_split = None
        self.gh = None
        self.h_length = None
        self.gh_length = None
        self.capacity = None

        # filled as training goes on
        self.root = None
        self.record = None  # saving split thresholds determined by active party
        self.update_pred = None  # saving tree raw output

    def fit(self, gradient, hessian, bin_index, bin_split):
        """single tree training, return raw output"""

        self.bin_index, self.bin_split = bin_index, bin_split

        sample_num = bin_index.shape[0]
        if self.sampling_method == "uniform":
            sample_tag = np.ones(sample_num, dtype=int)
        else:
            raise ValueError

        if self.task == "binary":
            self.gh = np.array([gradient, hessian]).T
            self.update_pred = np.zeros(sample_num, dtype=float)

            gh_int, self.h_length, self.gh_length = gh_packing(gradient, hessian, self.fix_point_precision)
            self.capacity = self.crypto_system.key_size // self.gh_length

            if self.crypto_type == Const.PLAIN:
                gh_send = gh_int
            elif self.crypto_type == Const.PAILLIER:
                gh_send = self.crypto_system.encrypt_vector(gh_int)
                gh_send = np.array(gh_send)
            else:
                raise NotImplementedError

        elif self.task == "multi":
            self.gh = np.stack((gradient, hessian), axis=2)
            self.update_pred = np.zeros((sample_num, self.n_labels), dtype=float)

            gh_compress, self.h_length, self.gh_length = gh_compress_multi(
                gradient, hessian, self.fix_point_precision, self.crypto_system.key_size
            )
            self.capacity = self.crypto_system.key_size // self.gh_length

            if self.crypto_type == Const.PAILLIER:
                shape = gh_compress.shape
                gh_send = self.crypto_system.encrypt_vector(gh_compress.flatten())
                gh_send = np.reshape(gh_send, shape)
            else:
                raise NotImplementedError

        else:
            raise ValueError("No such task label.")

        print("gh packed")
        self.messenger.send(wrap_message("gh", content=(gh_send, self.compress, self.capacity, self.gh_length)))

        # start building tree
        self.root = self._build_tree(sample_tag, current_depth=0)

        return self.update_pred

    def predict(self, x_test):
        """single tree predict"""

        y_pred = []
        for sampleID in range(x_test.shape[0]):
            x_sample = x_test[sampleID: sampleID + 1, :].flatten()
            score = self._predict_value(x_sample, sampleID, self.root)
            y_pred.append(score)

        return np.array(y_pred)

    def _build_tree(self, sample_tag, current_depth, hist_list=None):
        # split only when conditions meet
        if current_depth < self.max_depth - 1 and sample_tag.sum() >= self.min_split_samples:
            if hist_list is None:
                hist_list = self._get_hist_list(sample_tag)

            party_id, feature_id, split_id, max_gain = find_split(hist_list, self.task, self.reg_lambda)

            if max_gain > self.min_split_gain:
                if party_id == 0:
                    # split in active party
                    record_id, sample_tag_left = self._save_record(feature_id, split_id, sample_tag)
                    print(f"threshold saved in record_id: {record_id}")

                else:
                    # ask corresponding passive party to split
                    self.messenger.send(
                        wrap_message("record", content=(party_id, feature_id, split_id, sample_tag))
                    )
                    data = self.messenger.recv()
                    assert data["name"] == "record"

                    record_id, sample_tag_left = data["content"]

                sample_tag_right = sample_tag - sample_tag_left

                # choose the easy part to directly compute hist, other part can be computed by a simple subtract
                if sample_tag_left.sum() <= sample_tag_right.sum():
                    hist_list_left = self._get_hist_list(sample_tag_left)
                    hist_list_right = [current - left for current, left in zip(hist_list, hist_list_left)]
                else:
                    hist_list_right = self._get_hist_list(sample_tag_right)
                    hist_list_left = [current - right for current, right in zip(hist_list, hist_list_right)]

                left_node = self._build_tree(sample_tag_left, current_depth + 1, hist_list_left)
                right_node = self._build_tree(sample_tag_right, current_depth + 1, hist_list_right)

                return _DecisionNode(
                    party_id=party_id, record_id=record_id, left_branch=left_node, right_branch=right_node
                )

        # compute leaf weight
        if self.task == "multi":
            leaf_value = leaf_weight_multi(self.gh, sample_tag, self.reg_lambda)
            update_temp = np.dot(sample_tag.reshape(-1, 1), leaf_value.reshape(1, -1))
        else:
            leaf_value = leaf_weight(self.gh, sample_tag, self.reg_lambda)
            update_temp = np.dot(sample_tag, leaf_value)

        self.update_pred += update_temp

        return _DecisionNode(value=leaf_value)

    def _save_record(self, feature_id, split_id, sample_tag):
        record = np.array([feature_id, self.bin_split[feature_id][split_id]]).reshape(1, 2)

        if self.record is None:
            self.record = record
        else:
            self.record = np.concatenate((self.record, record), axis=0)

        record_id = len(self.record) - 1

        sample_tag_left = sample_tag.copy()  # avoid modification on sample_tag
        sample_tag_left[self.bin_index[:, feature_id].flatten() > split_id] = 0

        return record_id, sample_tag_left

    def _predict_value(self, x_sample, sample_id, tree_node: _DecisionNode = None):
        """predict a sample"""

        if tree_node is None:
            tree_node = self.root

        # If we have a value (i.e. we're at a leaf), return value as the prediction
        if tree_node.value is not None:
            return tree_node.value

        if tree_node.party_id == 0:
            # judge in active party
            feature_id = int(self.record[tree_node.record_id][0])
            threshold = self.record[tree_node.record_id][1]

            branch_tag = True if x_sample[feature_id] > threshold else False  # avoid numpy bool
        else:
            # ask corresponding passive party to judge
            self.messenger.send(wrap_message("judge", content=(tree_node.party_id, sample_id, tree_node.record_id)))
            data = self.messenger.recv()
            assert data["name"] == "judge"

            branch_tag = data["content"]

        # query in corresponding branch
        if branch_tag is True:
            return self._predict_value(x_sample, sample_id, tree_node.right_branch)
        else:
            return self._predict_value(x_sample, sample_id, tree_node.left_branch)

    def _get_hist_list(self, sample_tag):
        # 1. inform passive party to compute hist based on sample_tag
        self.messenger.send(wrap_message("hist", content=sample_tag))

        # 2. active party computes hist
        active_hist = ActiveHist.compute_hist(
            task=self.task, n_labels=self.n_labels, sample_tag=sample_tag, bin_index=self.bin_index, gh=self.gh
        )

        # 3. get passive party hist
        passive_hist = self._get_passive_hist()

        hist_list = [active_hist, passive_hist]

        return hist_list

    def _get_passive_hist(self):
        data = self.messenger.recv()
        assert data["name"] == "hist"

        if self.task == "multi":
            if self.crypto_type == Const.PAILLIER:
                bin_gh_compress_multi = data["content"]
                hist = ActiveHist.decompress_multi_hist(
                    task=self.task,
                    n_labels=self.n_labels,
                    capacity=self.capacity,
                    bin_gh_compress_multi=bin_gh_compress_multi,
                    h_length=self.h_length,
                    gh_length=self.gh_length,
                    r=self.fix_point_precision,
                    crypto_system=self.crypto_system,
                )

            else:
                raise ValueError("No such encoding type!")

        elif self.task == "binary":
            if self.crypto_type == Const.PLAIN:
                bin_gh_plain = data["content"]
                hist = ActiveHist.splitgh_hist(
                    task=self.task,
                    n_labels=self.n_labels,
                    bin_gh_int=bin_gh_plain,
                    h_length=self.h_length,
                    r=self.fix_point_precision,
                )

            elif self.crypto_type == Const.PAILLIER:
                if self.compress:
                    bin_gh_enc_compress = data["content"]
                    hist = ActiveHist.decompress_hist(
                        task=self.task,
                        n_labels=self.n_labels,
                        capacity=self.capacity,
                        bin_gh_compress=bin_gh_enc_compress,
                        h_length=self.h_length,
                        gh_length=self.gh_length,
                        r=self.fix_point_precision,
                        crypto_system=self.crypto_system,
                    )
                else:
                    bin_gh_enc = data["content"]
                    hist = ActiveHist.decrypt_hist(
                        task=self.task,
                        n_labels=self.n_labels,
                        bin_gh_enc=bin_gh_enc,
                        h_length=self.h_length,
                        r=self.fix_point_precision,
                        crypto_system=self.crypto_system,
                    )

            else:
                raise ValueError("No such encoding type!")

        else:
            raise ValueError("No such task label.")

        return hist
