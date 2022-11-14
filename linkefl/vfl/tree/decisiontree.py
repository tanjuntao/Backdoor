import queue
import random
import threading
import time

import numpy as np

from collections import defaultdict
from multiprocessing import Pool
from typing import List
from termcolor import colored

from linkefl.common.error import DisconnectedError
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.crypto.base import CryptoSystem
from linkefl.messenger.base import Messenger
from linkefl.vfl.tree.hist import ActiveHist
from linkefl.vfl.tree.data_functions import (
    wrap_message,
    random_sampling,
    goss_sampling,
)
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
        subsample: float = 1,
        top_rate: float = 0.5,
        other_rate: float = 0.5,
        colsample_bytree: float = 1,
        pool: Pool = None,
        drop_protection: bool = False,
        reconnect_ports: list = []
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
            sampling_method: uniform or goss
            subsample: sample sampling ratio
            top_rate: parameter for goss_sampling, head retention sample ratio
            other_rate: parameter for goss_sampling, proportion of remaining samples sampled
            colsample_bytree: tree-level feature sampling scale
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
        self.subsample = subsample
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.colsample_bytree = colsample_bytree
        self.pool = pool

        # record feature importance
        self.feature_importance_info = {
            "split": defaultdict(int),
            "gain": defaultdict(float),
            "cover": defaultdict(float)
        }
        self.messengers_validTag = [True for _ in range(len(self.messengers))]
        self.model_phase = "online_inference"
        self.drop_protection = drop_protection
        self.reconnect_ports = reconnect_ports

        # given when training starts
        self.bin_index_selected = None
        self.bin_split = None
        self.gh = None
        self.feature_index_selected = None
        self.h_length = None
        self.gh_length = None
        self.capacity = None

        # filled as training goes on
        self.root = None
        self.record = None  # saving split thresholds determined by active party
        self.update_pred = None  # saving tree raw output


    def fit(self, gradient, hessian, bin_index, bin_split):
        """single tree training, return raw output"""
        self.model_phase = "train"

        sample_num, feature_num = bin_index.shape[0], bin_index.shape[1]

        # tree-level samples sampling
        if self.sampling_method == "uniform":
            selected_g, selected_h, selected_idx = random_sampling(gradient, hessian, self.subsample)
        elif self.sampling_method == "goss":
            selected_g, selected_h, selected_idx = goss_sampling(gradient, hessian, self.top_rate, self.other_rate)
        else:
            raise ValueError

        # tree-level feature sampling
        self.feature_index_selected = random.sample(list(range(feature_num)), int(feature_num * self.colsample_bytree))
        self.feature_index_selected.sort()
        # reset bin_index
        self.bin_index_selected = np.array(bin_index.copy())
        self.bin_index_selected = self.bin_index_selected[:, self.feature_index_selected]
        # print(self.feature_index_selected, self.bin_index_selected)
        self.bin_split = bin_split
        # self.bin_index_selected, self.bin_split = feature_sampling(bin_index, bin_split, self.feature_index_selected)

        sample_tag_selected = np.zeros(sample_num, dtype=int)
        sample_tag_selected[selected_idx] = 1
        sample_tag_unselected = np.ones(sample_num, dtype=int) - sample_tag_selected

        # try to restore disconnect messengers
        for party_id, validTag in enumerate(self.messengers_validTag):
            if not validTag:
                self._reconnect_passiveParty(party_id)

        # Implementation logic with sampling
        if self.task == "binary":
            self.gh = np.array([gradient, hessian]).T
            self.update_pred = np.zeros(sample_num, dtype=float)

            selected_gh_int, self.h_length, self.gh_length = gh_packing(
                selected_g, selected_h, self.fix_point_precision
            )

            self.capacity = self.crypto_system.key_size // self.gh_length

            if self.crypto_type == Const.PLAIN:
                gh_send = np.zeros(sample_num)
                for i, idx in enumerate(selected_idx):
                    gh_send[idx] = selected_gh_int[i]
            elif self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
                gh_send = [0 for _ in range(sample_num)]
                selected_gh_enc = self.crypto_system.encrypt_data(selected_gh_int, self.pool)
                for i, idx in enumerate(selected_idx):
                    gh_send[idx] = selected_gh_enc[i]
                gh_send = np.array(gh_send)
            else:
                raise NotImplementedError

        elif self.task == "multi":
            self.gh = np.stack((gradient, hessian), axis=2)
            self.update_pred = np.zeros((sample_num, self.n_labels), dtype=float)

            selected_gh_compress, self.h_length, self.gh_length = gh_compress_multi(
                selected_g,
                selected_h,
                self.fix_point_precision,
                self.crypto_system.key_size,
            )
            self.capacity = self.crypto_system.key_size // self.gh_length

            # The number of ciphertexts occupied by each sample
            sample2enc_num = (self.n_labels + self.capacity - 1) // self.capacity
            gh_send = [[0 for _ in range(sample2enc_num)] for _ in range(sample_num)]

            if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
                selected_gh_enc = self.crypto_system.encrypt_data(selected_gh_compress, self.pool)
                for i, sample_idx in enumerate(selected_idx):
                    for j in range(sample2enc_num):
                        gh_send[sample_idx][j] = selected_gh_enc[i][j]
                gh_send = np.array(gh_send)
            else:
                raise NotImplementedError

        else:
            raise ValueError("No such task label.")

        self.logger.log("gh packed")

        while True:
            try:
                for i, messenger in enumerate(self.messengers):
                    if self.messengers_validTag[i]:
                        messenger.send(
                            wrap_message("gh", content=(gh_send, self.compress, self.capacity, self.gh_length))
                        )
                # start building tree
                self.root = self._build_tree(sample_tag_selected, sample_tag_unselected, current_depth=0)

            except DisconnectedError as e:
                self.logger.log(f"passive party {e.disconnect_party_id} is disconnected.")
                print(e)

                if e.disconnect_phase == 'hist':
                    # build histogram phase disconnected, need to clean up channels
                    for passive_party_id in range(e.disconnect_party_id+1, len(self.messengers)):
                        data, passive_party_connected = self.messengers[passive_party_id].recv()     # clear message
                        if not passive_party_connected:     # Multiple parties are disconnected at the same time
                            self._reconnect_passiveParty(passive_party_id)

                self._reconnect_passiveParty(e.disconnect_party_id)
            else:
                # if no exception occurs, break out of the loop
                break

        fit_result = {
            "update_pred": self.update_pred,
            "feature_importance_info": self.feature_importance_info
        }
        return fit_result

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
        sample_tag_selected,
        sample_tag_unselected,
        current_depth,
        *,
        hist_list=None,
        party_id=None,
        feature_id=None,
        split_id=None,
        max_gain=None
    ):
        # split only when conditions meet
        if (
            current_depth < self.max_depth - 1
            and sample_tag_selected.sum() >= self.min_split_samples
        ):
            if hist_list is None:
                # happens in root
                (
                    hist_list,
                    party_id,
                    feature_id,
                    split_id,
                    max_gain,
                ) = self._get_hist_list(sample_tag_selected)
            if party_id is None:
                # happens in sub hist
                party_id, feature_id, split_id, max_gain = find_split(
                    hist_list, self.task, self.reg_lambda
                )

            if max_gain > self.min_split_gain:
                if party_id == 0:
                    # split in active party
                    feature_id_origin, record_id, sample_tag_selected_left, sample_tag_unselected_left = self._save_record(
                        feature_id, split_id, sample_tag_selected, sample_tag_unselected
                    )
                    self.logger.log(f"threshold saved in record_id: {record_id}")

                else:
                    # ask corresponding passive party to split
                    self.messengers[party_id - 1].send(
                        wrap_message(
                            "record", content=(feature_id, split_id, sample_tag_selected, sample_tag_unselected)
                        )
                    )
                    if self.drop_protection:
                        data, passive_party_connected = self.messengers[party_id - 1].recv()
                        if not passive_party_connected:
                            raise DisconnectedError(disconnect_phase='record', disconnect_party_id=party_id-1)
                    else:
                        data = self.messengers[party_id - 1].recv()

                    # print(data)
                    assert data["name"] == "record"

                    # Get the selected feature index to the original feature index
                    feature_id_origin, record_id, sample_tag_selected_left , sample_tag_unselected_left = data["content"]

                # store feature split information
                self.feature_importance_info['split'][f'client{party_id}_feature{feature_id_origin}'] += 1
                self.feature_importance_info['gain'][f'client{party_id}_feature{feature_id_origin}'] += max_gain
                self.feature_importance_info['cover'][f'client{party_id}_feature{feature_id_origin}'] += sum(
                    sample_tag_selected)
                self.logger.log(f"store feature split information")

                sample_tag_selected_right = sample_tag_selected - sample_tag_selected_left
                sample_tag_unselected_right = sample_tag_unselected - sample_tag_unselected_left

                # choose the easy part to directly compute hist, the other part
                # can be computed by a simple subtract
                if sample_tag_selected_left.sum() <= sample_tag_selected_right.sum():
                    (
                        hist_list_left,
                        party_id_left,
                        feature_id_left,
                        split_id_left,
                        max_gain_left,
                    ) = self._get_hist_list(sample_tag_selected_left)
                    hist_list_right = [
                        current - left
                        for current, left in zip(hist_list, hist_list_left)
                    ]
                    left_node = self._build_tree(
                        sample_tag_selected_left,
                        sample_tag_unselected_left,
                        current_depth + 1,
                        hist_list=hist_list_left,
                        party_id=party_id_left,
                        feature_id=feature_id_left,
                        split_id=split_id_left,
                        max_gain=max_gain_left
                    )
                    right_node = self._build_tree(
                        sample_tag_selected_right,
                        sample_tag_unselected_right,
                        current_depth + 1,
                        hist_list=hist_list_right
                    )
                else:
                    (
                        hist_list_right,
                        party_id_right,
                        feature_id_right,
                        split_id_right,
                        max_gain_right,
                    ) = self._get_hist_list(sample_tag_selected_right)
                    hist_list_left = [
                        current - right
                        for current, right in zip(hist_list, hist_list_right)
                    ]
                    left_node = self._build_tree(
                        sample_tag_selected_left,
                        sample_tag_unselected_left,
                        current_depth + 1,
                        hist_list=hist_list_left
                    )
                    right_node = self._build_tree(
                        sample_tag_selected_right,
                        sample_tag_unselected_right,
                        current_depth + 1,
                        hist_list=hist_list_right,
                        party_id=party_id_right,
                        feature_id=feature_id_right,
                        split_id=split_id_right,
                        max_gain=max_gain_right
                    )

                return _DecisionNode(
                    party_id=party_id,
                    record_id=record_id,
                    left_branch=left_node,
                    right_branch=right_node,
                )

        # compute leaf weight
        if self.task == "multi":
            leaf_value = leaf_weight_multi(self.gh, sample_tag_selected, self.reg_lambda)
            update_temp = np.dot(sample_tag_selected.reshape(-1, 1), leaf_value.reshape(1, -1)) + \
                            np.dot(sample_tag_unselected.reshape(-1, 1), leaf_value.reshape(1, -1))
        else:
            leaf_value = leaf_weight(self.gh, sample_tag_selected, self.reg_lambda)
            update_temp = np.dot(sample_tag_selected, leaf_value) + np.dot(sample_tag_unselected, leaf_value)

        self.update_pred += update_temp

        return _DecisionNode(value=leaf_value)

    def _save_record(self, feature_id, split_id, sample_tag_selected, sample_tag_unselected):
        # Map the selected feature index to the original feature index
        feature_id_origin = self.feature_index_selected[feature_id]

        record = np.array([feature_id_origin, self.bin_split[feature_id_origin][split_id]]).reshape(
            1, 2
        )

        if self.record is None:
            self.record = record
        else:
            self.record = np.concatenate((self.record, record), axis=0)

        record_id = len(self.record) - 1

        sample_tag_selected_left = np.array(sample_tag_selected.copy())  # avoid modification on sample_tag_selected
        sample_tag_selected_left[self.bin_index_selected[:, feature_id].flatten() > split_id] = 0

        sample_tag_unselected_left = np.array(sample_tag_unselected.copy())
        sample_tag_unselected_left[self.bin_index_selected[:, feature_id].flatten() > split_id] = 0

        return feature_id_origin, record_id, sample_tag_selected_left, sample_tag_unselected_left

    def _predict_value(self, x_sample, sample_id,
                       tree_node: _DecisionNode = None):
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

            if self.drop_protection:
                data, passive_party_connected = self.messengers[tree_node.party_id - 1].recv()
                if not passive_party_connected:
                    if self.model_phase == "train":
                        raise DisconnectedError(
                            disconnect_phase=f"predict_{self.model_phase}", disconnect_party_id=tree_node.party_id - 1)
                    else:
                        raise RuntimeError(f"Passive party {tree_node.party_id - 1} disconnect.")
            else:
                data = self.messengers[tree_node.party_id - 1].recv()

            assert data["name"] == "judge"

            branch_tag = data["content"]

        # query in corresponding branch
        if branch_tag is True:
            return self._predict_value(x_sample, sample_id, tree_node.right_branch)
        else:
            return self._predict_value(x_sample, sample_id, tree_node.left_branch)

    def _get_hist_list(self, sample_tag):
        q = queue.Queue()

        # 1. inform passive party to compute hist based on sample_tag
        for i, messenger in enumerate(self.messengers):
            if self.messengers_validTag[i]:
                messenger.send(wrap_message("hist", content=sample_tag))

        # 2. get passive party hist
        thread_list = []
        for i, messenger in enumerate(self.messengers, 1):
            if not self.messengers_validTag[i-1]:
                continue

            passive_hist = self._get_passive_hist(messenger=messenger, messenger_id=i-1)
            t = threading.Thread(
                target=self._process_passive_hist, args=(passive_hist, i, q)
            )
            t.start()
            thread_list.append(t)

        # 3. active party computes hist
        active_hist = ActiveHist.compute_hist(
            task=self.task,
            n_labels=self.n_labels,
            sample_tag=sample_tag,
            bin_index=self.bin_index_selected,
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

    def _process_passive_hist(self, passive_hist, i, q: queue.Queue):
        _, feature_id, split_id, max_gain = find_split(
            [passive_hist], self.task, self.reg_lambda
        )
        q.put((i, passive_hist, feature_id, split_id, max_gain))

    def _get_passive_hist(self, messenger, messenger_id):
        if self.drop_protection:
            data, passive_party_connected = messenger.recv()
            if not passive_party_connected:
                raise DisconnectedError(disconnect_phase='hist',  disconnect_party_id=messenger_id)
        else:
            data = messenger.recv()

        assert data["name"] == "hist"

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

    def _reconnect_passiveParty(self, disconnect_party_id):
        self.logger.log(f"start reconnect to passive party {disconnect_party_id}")
        is_reconnect, reconnect_count = False, 1
        reconnect_max_count, reconnect_gap = 10, 20

        while not is_reconnect:
            if reconnect_count > reconnect_max_count:
                break

            print(colored(f"try to reconnect, reconnect count : {reconnect_count}", "green"))
            is_reconnect = self.messengers[disconnect_party_id].try_reconnect(self.reconnect_ports[disconnect_party_id])
            reconnect_count += 1
            time.sleep(reconnect_gap)

        if is_reconnect:
            print(colored(f"reconnect success.", "red"))
            self.logger.log("reconnect success")
            self.messengers_validTag[disconnect_party_id] = True
        else:
            # drop disconnected party
            print(colored(f"reconnect failed.", "red"))
            self.logger.log("reconnect failed")
            self.messengers_validTag[disconnect_party_id] = False
