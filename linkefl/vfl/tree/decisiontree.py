import queue
import random
import threading
import time
from collections import defaultdict, deque
from multiprocessing import Pool
from typing import List

import numpy as np
from termcolor import colored

from linkefl.base import BaseCryptoSystem, BaseMessenger
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.vfl.tree.data_functions import goss_sampling, random_sampling, wrap_message
from linkefl.vfl.tree.error import DisconnectedError
from linkefl.vfl.tree.exception_thread import ExcThread
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
        hist_list=None,
        sample_tag_selected=None,
        sample_tag_unselected=None,
        split_party_id=None,
        split_feature_id=None,
        split_bin_id=None,
        split_gain=None,
        depth=None,
    ):
        # intermediate node
        self.party_id = party_id
        self.record_id = record_id
        self.left_branch = left_branch
        self.right_branch = right_branch

        # leaf node
        self.value = value

        # training data
        self.hist_list = hist_list
        self.sample_tag_selected = sample_tag_selected
        self.sample_tag_unselected = sample_tag_unselected
        self.split_party_id = split_party_id
        self.split_feature_id = split_feature_id
        self.split_bin_id = split_bin_id
        self.split_gain = split_gain
        self.depth = depth

    def __lt__(self, other):
        return self.split_gain > other.split_gain


class DecisionTree:
    def __init__(
        self,
        task: str,
        n_labels: int,
        crypto_type,
        crypto_system,
        messengers: List[BaseMessenger],
        logger: GlobalLogger,
        *,
        training_mode: str = "lightgbm",
        compress: bool = False,
        max_depth: int = 4,
        max_num_leaves: int = 31,
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
        reconnect_ports: List[int] = None,
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
            other_rate: parameter for goss_sampling,
                proportion of remaining samples sampled
            colsample_bytree: tree-level feature sampling scale
            pool: multiprocessing pool
        """

        self.task = task
        self.n_labels = n_labels
        self.crypto_type = crypto_type
        self.crypto_system = crypto_system
        self.messengers = messengers
        self.logger = logger

        self.training_mode = training_mode
        self.compress = compress
        self.max_depth = max_depth
        self.max_num_leaves = max_num_leaves
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
            "cover": defaultdict(float),
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
        self.messengers_recvTag = None

    def fit(self, gradient, hessian, bin_index, bin_split):
        # 0. prepare: get params and reconnect
        self.model_phase = "train"
        sample_num, feature_num = bin_index.shape[0], bin_index.shape[1]

        if self.task == "regression" or self.task == "binary":
            self.gh = np.array([gradient, hessian]).T
            self.update_pred = np.zeros(sample_num, dtype=float)
        else:
            self.gh = np.stack((gradient, hessian), axis=2)
            self.update_pred = np.zeros((sample_num, self.n_labels), dtype=float)

        for party_id, validTag in enumerate(self.messengers_validTag):
            if not validTag:
                self._reconnect_passiveParty(party_id)

        # 1. sample: sample sampling and feature sampling
        if self.sampling_method == "uniform":
            selected_g, selected_h, selected_idx = random_sampling(
                gradient, hessian, self.subsample
            )
        elif self.sampling_method == "goss":
            selected_g, selected_h, selected_idx = goss_sampling(
                gradient, hessian, self.top_rate, self.other_rate
            )
        else:
            raise ValueError

        self.feature_index_selected = random.sample(
            list(range(feature_num)), int(feature_num * self.colsample_bytree)
        )
        self.feature_index_selected.sort()
        self.bin_index_selected = np.array(bin_index.copy())
        self.bin_index_selected = self.bin_index_selected[
            :, self.feature_index_selected
        ]
        self.bin_split = bin_split

        # sample_tag: selected training sample, full_sample_tag: all sample
        sample_tag_selected = np.zeros(sample_num, dtype=int)
        sample_tag_selected[selected_idx] = 1
        sample_tag_unselected = np.ones(sample_num, dtype=int) - sample_tag_selected

        # 2. gh packing
        gh_send = self._get_gh_send(sample_num, selected_g, selected_h, selected_idx)

        # 3. start train
        while True:
            try:
                for i, messenger in enumerate(self.messengers):
                    if self.messengers_validTag[i]:
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
                if self.training_mode == "lightgbm":
                    self.root = self._build_tree_lightgbm(
                        sample_tag_selected, sample_tag_unselected
                    )
                elif self.training_mode == "xgboost":
                    self.root = self._build_tree_xgb(
                        sample_tag_selected, sample_tag_unselected
                    )
                else:
                    raise NotImplementedError("Unsupported training mode.")

            except DisconnectedError as e:
                self.logger.log(
                    f"passive party {e.disconnect_party_id} is disconnected."
                )
                print(e)

                if e.disconnect_phase == "hist":
                    # build histogram phase disconnected, need to clean up channels
                    for passive_party_id, recv_tag in enumerate(
                        self.messengers_recvTag
                    ):
                        if recv_tag or passive_party_id == e.disconnect_party_id:
                            continue
                        data, passive_party_connected = self.messengers[
                            passive_party_id
                        ].recv()  # clear message
                        if (
                            not passive_party_connected
                        ):  # Multiple parties are disconnected at the same time
                            self._reconnect_passiveParty(passive_party_id)

                self._reconnect_passiveParty(e.disconnect_party_id)
            else:
                # if no exception occurs, break out of the loop
                break

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

    def _build_tree_xgb(self, sample_tag_selected, sample_tag_unselected):
        root = self._gen_node(sample_tag_selected, sample_tag_unselected, depth=0)

        split_node_candidates = deque()
        split_node_candidates.append(root)

        while len(split_node_candidates) > 0:
            node = split_node_candidates.popleft()
            if (
                node.depth < self.max_depth
                and node.sample_tag_selected.sum() >= self.min_split_samples
                and node.split_gain > self.min_split_gain
            ):
                (
                    feature_id_origin,
                    record_id,
                    sample_tag_selected_left,
                    sample_tag_unselected_left,
                ) = self._get_split_info(node)

                left_node, right_node = self._gen_child_node(
                    node, sample_tag_selected_left, sample_tag_unselected_left
                )
                split_node_candidates.append(left_node)
                split_node_candidates.append(right_node)

                node.party_id = node.split_party_id
                node.record_id = record_id
                node.left_branch = left_node
                node.right_branch = right_node

                # store feature split information
                self.feature_importance_info["split"][
                    f"client{node.party_id}_feature{feature_id_origin}"
                ] += 1
                self.feature_importance_info["gain"][
                    f"client{node.party_id}_feature{feature_id_origin}"
                ] += node.split_gain
                self.feature_importance_info["cover"][
                    f"client{node.party_id}_feature{feature_id_origin}"
                ] += sum(node.sample_tag_selected)
                self.logger.log("store feature split information")

            else:
                # compute leaf weight
                if self.task == "multi":
                    leaf_value = leaf_weight_multi(
                        self.gh, node.sample_tag_selected, self.reg_lambda
                    )
                    update_temp = np.dot(
                        node.sample_tag_selected.reshape(-1, 1),
                        leaf_value.reshape(1, -1),
                    ) + np.dot(
                        node.sample_tag_unselected.reshape(-1, 1),
                        leaf_value.reshape(1, -1),
                    )
                else:
                    leaf_value = leaf_weight(
                        self.gh, node.sample_tag_selected, self.reg_lambda
                    )
                    update_temp = np.dot(node.sample_tag_selected, leaf_value) + np.dot(
                        node.sample_tag_unselected, leaf_value
                    )

                node.value = leaf_value
                self.update_pred += update_temp

        return root

    def _build_tree_lightgbm(self, sample_tag_selected, sample_tag_unselected):
        root = self._gen_node(sample_tag_selected, sample_tag_unselected, depth=0)
        split_node_candidates = queue.PriorityQueue()
        split_node_candidates.put(root)

        num_leaves = 0
        while (
            not split_node_candidates.empty()
        ) and num_leaves + split_node_candidates.qsize() < self.max_num_leaves:
            node = split_node_candidates.get()
            if (
                node.depth < self.max_depth
                and node.sample_tag_selected.sum() >= self.min_split_samples
                and node.split_gain > self.min_split_gain
                and num_leaves + split_node_candidates.qsize() < self.max_num_leaves
            ):
                (
                    feature_id_origin,
                    record_id,
                    sample_tag_selected_left,
                    sample_tag_unselected_left,
                ) = self._get_split_info(node)

                left_node, right_node = self._gen_child_node(
                    node, sample_tag_selected_left, sample_tag_unselected_left
                )
                split_node_candidates.put(left_node)
                split_node_candidates.put(right_node)

                node.party_id = node.split_party_id
                node.record_id = record_id
                node.left_branch = left_node
                node.right_branch = right_node

                # store feature split information
                self.feature_importance_info["split"][
                    f"client{node.party_id}_feature{feature_id_origin}"
                ] += 1
                self.feature_importance_info["gain"][
                    f"client{node.party_id}_feature{feature_id_origin}"
                ] += node.split_gain
                self.feature_importance_info["cover"][
                    f"client{node.party_id}_feature{feature_id_origin}"
                ] += sum(node.sample_tag_selected)
                self.logger.log("store feature split information")

            else:
                # compute leaf weight
                if self.task == "multi":
                    leaf_value = leaf_weight_multi(
                        self.gh, node.sample_tag_selected, self.reg_lambda
                    )
                    update_temp = np.dot(
                        node.sample_tag_selected.reshape(-1, 1),
                        leaf_value.reshape(1, -1),
                    ) + np.dot(
                        node.sample_tag_unselected.reshape(-1, 1),
                        leaf_value.reshape(1, -1),
                    )
                else:
                    leaf_value = leaf_weight(
                        self.gh, node.sample_tag_selected, self.reg_lambda
                    )
                    update_temp = np.dot(node.sample_tag_selected, leaf_value) + np.dot(
                        node.sample_tag_unselected, leaf_value
                    )
                num_leaves += 1
                node.value = leaf_value
                self.update_pred += update_temp

        while not split_node_candidates.empty():
            node = split_node_candidates.get()
            # compute leaf weight
            if self.task == "multi":
                leaf_value = leaf_weight_multi(
                    self.gh, node.sample_tag_selected, self.reg_lambda
                )
                update_temp = np.dot(
                    node.sample_tag_selected.reshape(-1, 1), leaf_value.reshape(1, -1)
                ) + np.dot(
                    node.sample_tag_unselected.reshape(-1, 1), leaf_value.reshape(1, -1)
                )
            else:
                leaf_value = leaf_weight(
                    self.gh, node.sample_tag_selected, self.reg_lambda
                )
                update_temp = np.dot(node.sample_tag_selected, leaf_value) + np.dot(
                    node.sample_tag_unselected, leaf_value
                )
            num_leaves += 1
            node.value = leaf_value
            self.update_pred += update_temp

        return root

    def _get_gh_send(self, sample_num, selected_g, selected_h, selected_idx):
        if self.task == "regression" or self.task == "binary":
            selected_gh_int, self.h_length, self.gh_length = gh_packing(
                selected_g, selected_h, self.fix_point_precision
            )

            self.capacity = self.crypto_system.key_size // self.gh_length

            if self.crypto_type == Const.PLAIN:
                gh_send = np.zeros(sample_num, dtype=object)
                for i, idx in enumerate(selected_idx):
                    gh_send[idx] = selected_gh_int[i]
            elif self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
                gh_send = [0 for _ in range(sample_num)]
                selected_gh_enc = self.crypto_system.encrypt_data(
                    selected_gh_int, pool=self.pool
                )
                for i, idx in enumerate(selected_idx):
                    gh_send[idx] = selected_gh_enc[i]
                gh_send = np.array(gh_send, dtype=object)
            else:
                raise NotImplementedError

        elif self.task == "multi":
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
                selected_gh_enc = self.crypto_system.encrypt_data(
                    selected_gh_compress, pool=self.pool
                )
                for i, sample_idx in enumerate(selected_idx):
                    for j in range(sample2enc_num):
                        gh_send[sample_idx][j] = selected_gh_enc[i][j]
                gh_send = np.array(gh_send)
            else:
                raise NotImplementedError

        else:
            raise ValueError("No such task label.")

        self.logger.log("gh packed.")
        return gh_send

    def _gen_child_node(
        self, node, sample_tag_selected_left, sample_tag_unselected_left
    ):
        sample_tag_selected_right = node.sample_tag_selected - sample_tag_selected_left
        sample_tag_unselected_right = (
            node.sample_tag_unselected - sample_tag_unselected_left
        )

        if sum(sample_tag_selected_left) < sum(sample_tag_selected_right):
            left_node = self._gen_node(
                sample_tag_selected_left,
                sample_tag_unselected_left,
                depth=node.depth + 1,
            )
            hist_list_right = [
                None if current is None or left is None else current - left
                for current, left in zip(node.hist_list, left_node.hist_list)
            ]
            right_node = self._gen_node(
                sample_tag_selected_right,
                sample_tag_unselected_right,
                depth=node.depth + 1,
                hist_list=hist_list_right,
            )
        else:
            right_node = self._gen_node(
                sample_tag_selected_right,
                sample_tag_unselected_right,
                depth=node.depth + 1,
            )
            hist_list_left = [
                None if current is None or right is None else current - right
                for current, right in zip(node.hist_list, right_node.hist_list)
            ]
            left_node = self._gen_node(
                sample_tag_selected_left,
                sample_tag_unselected_left,
                depth=node.depth + 1,
                hist_list=hist_list_left,
            )

        return left_node, right_node

    def _gen_node(
        self, sample_tag_selected, sample_tag_unselected, depth=0, hist_list=None
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
        else:
            # happens in sub hist
            party_id, feature_id, split_id, max_gain = find_split(
                hist_list, self.task, self.reg_lambda
            )

        return _DecisionNode(
            hist_list=hist_list,
            sample_tag_selected=sample_tag_selected,
            sample_tag_unselected=sample_tag_unselected,
            split_party_id=party_id,
            split_feature_id=feature_id,
            split_bin_id=split_id,
            split_gain=max_gain,
            depth=depth,
        )

    def _get_split_info(self, node):
        if node.split_party_id == 0:
            # split in active party
            (
                feature_id_origin,
                record_id,
                sample_tag_selected_left,
                sample_tag_unselected_left,
            ) = self._save_record(
                node.split_feature_id,
                node.split_bin_id,
                node.sample_tag_selected,
                node.sample_tag_unselected,
            )
            self.logger.log(f"threshold saved in record_id: {record_id}")
        else:
            # ask corresponding passive party to split
            self.messengers[node.split_party_id - 1].send(
                wrap_message(
                    "record",
                    content=(
                        node.split_feature_id,
                        node.split_bin_id,
                        node.sample_tag_selected,
                        node.sample_tag_unselected,
                    ),
                )
            )
            if self.drop_protection:
                data, passive_party_connected = self.messengers[
                    node.split_party_id - 1
                ].recv()
                if not passive_party_connected:
                    raise DisconnectedError(
                        disconnect_phase="record",
                        disconnect_party_id=node.split_party_id - 1,
                    )
            else:
                data = self.messengers[node.split_party_id - 1].recv()

            # print(data)
            assert data["name"] == "record"
            # Get the selected feature index to the original feature index
            (
                feature_id_origin,
                record_id,
                sample_tag_selected_left,
                sample_tag_unselected_left,
            ) = data["content"]

        return (
            feature_id_origin,
            record_id,
            sample_tag_selected_left,
            sample_tag_unselected_left,
        )

    def _save_record(
        self, feature_id, split_id, sample_tag_selected, sample_tag_unselected
    ):
        # Map the selected feature index to the original feature index
        feature_id_origin = self.feature_index_selected[feature_id]

        record = np.array(
            [feature_id_origin, self.bin_split[feature_id_origin][split_id]]
        ).reshape(1, 2)

        if self.record is None:
            self.record = record
        else:
            self.record = np.concatenate((self.record, record), axis=0)

        record_id = len(self.record) - 1

        sample_tag_selected_left = np.array(
            sample_tag_selected.copy()
        )  # avoid modification on sample_tag_selected
        sample_tag_selected_left[
            self.bin_index_selected[:, feature_id].flatten() > split_id
        ] = 0

        sample_tag_unselected_left = np.array(sample_tag_unselected.copy())
        sample_tag_unselected_left[
            self.bin_index_selected[:, feature_id].flatten() > split_id
        ] = 0

        return (
            feature_id_origin,
            record_id,
            sample_tag_selected_left,
            sample_tag_unselected_left,
        )

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

            if self.drop_protection:
                data, passive_party_connected = self.messengers[
                    tree_node.party_id - 1
                ].recv()
                if not passive_party_connected:
                    if self.model_phase == "train":
                        raise DisconnectedError(
                            disconnect_phase=f"predict_{self.model_phase}",
                            disconnect_party_id=tree_node.party_id - 1,
                        )
                    else:
                        raise RuntimeError(
                            f"Passive party {tree_node.party_id - 1} disconnect."
                        )
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
        self.messengers_recvTag = [False for _ in range(len(self.messengers))]
        self.mutex = threading.Lock()
        try:
            thread_list = []
            for i, messenger in enumerate(self.messengers, 1):
                if not self.messengers_validTag[i - 1]:
                    continue

                t = ExcThread(target=self._process_passive_hist, args=(messenger, i, q))
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

                for t in thread_list:
                    t.join()

        except DisconnectedError as e:
            self.mutex.release()  # free mutex
            raise e

        # 4. get the best gain
        best = [None] * 4
        hist_list = [None] * (1 + len(self.messengers))
        while not q.empty():
            party_id, hist, feature_id, split_id, max_gain = q.get()
            hist_list[party_id] = hist
            if best[3] is None or best[3] < max_gain:
                best = [party_id, feature_id, split_id, max_gain]

        return hist_list, best[0], best[1], best[2], best[3]

    def _process_passive_hist(self, messenger, i, q: queue.Queue):
        self.mutex.acquire()  # ensure that only one process is reading at a time

        if self.drop_protection:
            data, passive_party_connected = messenger.recv()
            if not passive_party_connected:
                self.disconnect_tag = True
                raise DisconnectedError(
                    disconnect_phase="hist", disconnect_party_id=i - 1
                )
        else:
            data = messenger.recv()

        assert data["name"] == "hist"
        self.messengers_recvTag[i - 1] = True

        self.mutex.release()

        passive_hist = self._get_passive_hist(data)
        _, feature_id, split_id, max_gain = find_split(
            [passive_hist], self.task, self.reg_lambda
        )
        q.put((i, passive_hist, feature_id, split_id, max_gain))

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

        elif self.task == "binary" or self.task == "regression":
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
        reconnect_max_count, reconnect_gap = 3, 5

        while not is_reconnect:
            if reconnect_count > reconnect_max_count:
                break

            print(
                colored(
                    f"try to reconnect, reconnect count : {reconnect_count}", "green"
                )
            )
            is_reconnect = self.messengers[disconnect_party_id].try_reconnect(
                self.reconnect_ports[disconnect_party_id]
            )
            reconnect_count += 1
            time.sleep(reconnect_gap)

        if is_reconnect:
            print(
                colored(
                    f"reconnect to passive_party_{disconnect_party_id} success.", "red"
                )
            )
            self.logger.log(f"reconnect to passive_party_{disconnect_party_id} success")
            self.messengers_validTag[disconnect_party_id] = True
        else:
            # drop disconnected party
            print(
                colored(
                    f"reconnect to passive_party_{disconnect_party_id} failed.", "red"
                )
            )
            self.logger.log(f"reconnect to passive_party_{disconnect_party_id} failed")
            self.messengers_validTag[disconnect_party_id] = False
