import queue
import threading
from multiprocessing import Pool
from typing import List

import numpy as np

from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.crypto.base import CryptoSystem
from linkefl.messenger.base import Messenger
from linkefl.vfl.tree.data_functions import wrap_message
from linkefl.vfl.tree.hist import ActiveHist
from linkefl.vfl.tree.train_functions import (
    find_split,
    gh_compress_multi,
    gh_packing,
    leaf_weight,
    leaf_weight_multi,
)


class _DecisionNode:
    def __init__(
        self,
        party_id=None,
        record_id=None,
        left_branch=None,
        right_branch=None,
        value=None,
    ):
        # intermediate node
        self.party_id = party_id
        self.record_id = record_id
        self.left_branch = left_branch
        self.right_branch = right_branch

        # leaf node
        self.value = value


class DecisionTree:
    def __init__(
        self,
        task: str,
        n_labels: int,
        crypto_type: str,
        crypto_system: CryptoSystem,
        messengers: List[Messenger],
        logger: GlobalLogger,
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
            fix_point_precision: binary length to keep when casting float to int
            sampling_method: uniform
            pool: multiprocessing pool
        """

        self.task = task
        self.n_labels = n_labels
        self.crypto_type = crypto_type
        self.crypto_system = crypto_system
        self.messengers = messengers
        self.logger = logger

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

    def fit(self, gradient, hessian, bin_index, bin_split, feature_importance_info):
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

            gh_int, self.h_length, self.gh_length = gh_packing(
                gradient, hessian, self.fix_point_precision
            )
            self.capacity = self.crypto_system.key_size // self.gh_length

            if self.crypto_type == Const.PLAIN:
                gh_send = gh_int
            elif self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
                gh_send = self.crypto_system.encrypt_data(gh_int, self.pool)
            else:
                raise NotImplementedError

        elif self.task == "multi":
            self.gh = np.stack((gradient, hessian), axis=2)
            self.update_pred = np.zeros((sample_num, self.n_labels), dtype=float)

            gh_compress, self.h_length, self.gh_length = gh_compress_multi(
                gradient,
                hessian,
                self.fix_point_precision,
                self.crypto_system.key_size,
            )
            self.capacity = self.crypto_system.key_size // self.gh_length

            if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
                gh_send = self.crypto_system.encrypt_data(gh_compress, self.pool)
            else:
                raise NotImplementedError

        else:
            raise ValueError("No such task label.")

        self.logger.log("gh packed")
        for messenger in self.messengers:
            messenger.send(
                wrap_message(
                    "gh",
                    content=(
                        gh_send,
                        self.compress,
                        self.capacity,
                        self.gh_length,
                    ),
                )
            )

        # start building tree
        self.root = self._build_tree(sample_tag, feature_importance_info, current_depth=0)

        return self.update_pred

    def predict(self, x_test):
        """single tree predict"""

        if self.root is None:
            return None

        y_pred = []
        for sampleID in range(x_test.shape[0]):
            x_sample = x_test[sampleID : sampleID + 1, :].flatten()
            score = self._predict_value(x_sample, sampleID, self.root)
            y_pred.append(score)

        return np.array(y_pred)

    def _build_tree(
        self,
        sample_tag,
        current_depth,
        feature_importance_info,
        *,
        hist_list=None,
        party_id=None,
        feature_id=None,
        split_id=None,
        max_gain=None,
    ):
        # split only when conditions meet
        if (
            current_depth < self.max_depth - 1
            and sample_tag.sum() >= self.min_split_samples
        ):
            if hist_list is None:
                # happens in root
                (
                    hist_list,
                    party_id,
                    feature_id,
                    split_id,
                    max_gain,
                ) = self._get_hist_list(sample_tag)
            if party_id is None:
                # happens in sub hist
                party_id, feature_id, split_id, max_gain = find_split(
                    hist_list, self.task, self.reg_lambda
                )

            if max_gain > self.min_split_gain:
                # store feature split information
                party = 'active party' if party_id == 0 else f'passive party {party_id}'
                feature_importance_info['split'][(party, f'feature {feature_id}')] += 1
                feature_importance_info['gain'][(party, f'feature {feature_id}')] += max_gain
                feature_importance_info['cover'][(party, f'feature {feature_id}')] += sum(sample_tag)

                if party_id == 0:
                    # split in active party
                    record_id, sample_tag_left = self._save_record(
                        feature_id, split_id, sample_tag
                    )
                    self.logger.log(f"threshold saved in record_id: {record_id}")

                else:
                    # ask corresponding passive party to split
                    self.messengers[party_id - 1].send(
                        wrap_message(
                            "record", content=(feature_id, split_id, sample_tag)
                        )
                    )
                    data = self.messengers[party_id - 1].recv()
                    assert data["name"] == "record"

                    record_id, sample_tag_left = data["content"]

                sample_tag_right = sample_tag - sample_tag_left

                # choose the easy part to directly compute hist, the other part
                # can be computed by a simple subtract
                if sample_tag_left.sum() <= sample_tag_right.sum():
                    (
                        hist_list_left,
                        party_id_left,
                        feature_id_left,
                        split_id_left,
                        max_gain_left,
                    ) = self._get_hist_list(sample_tag_left)
                    hist_list_right = [
                        current - left
                        for current, left in zip(hist_list, hist_list_left)
                    ]
                    left_node = self._build_tree(
                        sample_tag_left,
                        current_depth + 1,
                        hist_list=hist_list_left,
                        party_id=party_id_left,
                        feature_id=feature_id_left,
                        split_id=split_id_left,
                        max_gain=max_gain_left,
                    )
                    right_node = self._build_tree(
                        sample_tag_right,
                        current_depth + 1,
                        hist_list=hist_list_right,
                    )
                else:
                    (
                        hist_list_right,
                        party_id_right,
                        feature_id_right,
                        split_id_right,
                        max_gain_right,
                    ) = self._get_hist_list(sample_tag_right)
                    hist_list_left = [
                        current - right
                        for current, right in zip(hist_list, hist_list_right)
                    ]
                    left_node = self._build_tree(
                        sample_tag_left,
                        current_depth + 1,
                        hist_list=hist_list_left,
                    )
                    right_node = self._build_tree(
                        sample_tag_right,
                        current_depth + 1,
                        hist_list=hist_list_right,
                        party_id=party_id_right,
                        feature_id=feature_id_right,
                        split_id=split_id_right,
                        max_gain=max_gain_right,
                    )

                return _DecisionNode(
                    party_id=party_id,
                    record_id=record_id,
                    left_branch=left_node,
                    right_branch=right_node,
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
        record = np.array([feature_id, self.bin_split[feature_id][split_id]]).reshape(
            1, 2
        )

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

        # If we have a value (i.e. we're at a leaf), return value as prediction
        if tree_node.value is not None:
            return tree_node.value

        if tree_node.party_id == 0:
            # judge in active party
            feature_id = int(self.record[tree_node.record_id][0])
            threshold = self.record[tree_node.record_id][1]

            branch_tag = (
                True if x_sample[feature_id] > threshold else False
            )  # avoid numpy bool
        else:
            # ask corresponding passive party to judge
            self.messengers[tree_node.party_id - 1].send(
                wrap_message("judge", content=(sample_id, tree_node.record_id))
            )
            data = self.messengers[tree_node.party_id - 1].recv()
            assert data["name"] == "judge"

            branch_tag = data["content"]

        # query in corresponding branch
        if branch_tag is True:
            return self._predict_value(x_sample, sample_id, tree_node.right_branch)
        else:
            return self._predict_value(x_sample, sample_id, tree_node.left_branch)

    def _process_passive_hist(self, messenger, i, q: queue.Queue):
        data = messenger.recv()
        assert data["name"] == "hist"

        passive_hist = self._get_passive_hist(data)
        _, feature_id, split_id, max_gain = find_split(
            [passive_hist], self.task, self.reg_lambda
        )
        q.put((i, passive_hist, feature_id, split_id, max_gain))

    def _get_hist_list(self, sample_tag):
        q = queue.Queue()

        # 1. inform passive party to compute hist based on sample_tag
        for messenger in self.messengers:
            messenger.send(wrap_message("hist", content=sample_tag))

        # 2. get passive party hist
        thread_list = []
        for i, messenger in enumerate(self.messengers, 1):
            t = threading.Thread(
                target=self._process_passive_hist, args=(messenger, i, q)
            )
            t.start()
            thread_list.append(t)

        # 3. active party computes hist
        active_hist = ActiveHist.compute_hist(
            task=self.task,
            n_labels=self.n_labels,
            sample_tag=sample_tag,
            bin_index=self.bin_index,
            gh=self.gh,
        )
        _, feature_id, split_id, max_gain = find_split(
            [active_hist], self.task, self.reg_lambda
        )
        q.put((0, active_hist, feature_id, split_id, max_gain))

        # 4. get the best gain
        best = [None] * 4
        hist_list = [None] * (1 + len(self.messengers))
        for i in range(1 + len(self.messengers)):
            party_id, hist, feature_id, split_id, max_gain = q.get()
            hist_list[party_id] = hist
            if best[3] is None or best[3] < max_gain:
                best = [party_id, feature_id, split_id, max_gain]

        return hist_list, best[0], best[1], best[2], best[3]

    def _get_passive_hist(self, data):
        if self.task == "multi":
            if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
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
                    pool=self.pool,
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

            elif self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
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
                        pool=self.pool,
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
                        pool=self.pool,
                    )

            else:
                raise ValueError("No such encoding type!")

        else:
            raise ValueError("No such task label.")

        return hist
