import time
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.crypto.base import CryptoSystem
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import parse_label
from linkefl.messenger.base import Messenger
from linkefl.util import sigmoid
from linkefl.vfl.tree import DecisionTree
from linkefl.vfl.tree.data_functions import get_bin_info, wrap_message
from linkefl.vfl.tree.loss_functions import CrossEntropyLoss, MultiCrossEntropyLoss


class ActiveTreeParty:
    def __init__(
        self,
        n_trees: int,
        task: str,
        n_labels: int,
        crypto_type: str,
        crypto_system: CryptoSystem,
        messengers: list[Messenger],
        *,
        learning_rate: float = 0.3,
        compress: bool = False,
        max_bin: int = 16,
        max_depth: int = 4,
        reg_lambda: float = 0.1,
        min_split_samples: int = 3,
        min_split_gain: float = 1e-7,
        fix_point_precision: int = 53,
        sampling_method: str = "uniform",
        n_processes: int = 1,
    ):
        """Active Tree Party class to train and validate dataset

        Args:
            n_trees: number of trees
            task: binary or multi
            n_labels: number of labels, should be 2 if task is set as binary
            compress: can only be enabled when task is set as binary
            max_bin: max bin number for a feature point
            max_depth: max depth of a tree, including root
            reg_lambda: used to compute gain and leaf weight
            min_split_samples: minimum samples required to split
            min_split_gain: minimum gain required to split
            fix_point_precision: bit length to preserve when converting float to int
            sampling_method: uniform
            n_processes: number of processes in multiprocessing
        """

        self._check_parameters(task, n_labels, compress, sampling_method)

        self.task = task
        self.n_labels = n_labels
        self.messengers = messengers

        self.learning_rate = learning_rate
        self.max_bin = max_bin

        if n_processes > 1:
            self.pool = Pool(n_processes)
        else:
            self.pool = None

        # 初始化 loss
        if task == "binary":
            self.loss = CrossEntropyLoss()
        elif task == "multi":
            self.loss = MultiCrossEntropyLoss()
        else:
            raise ValueError("No such task label.")

        self.trees = [
            DecisionTree(
                task=task,
                n_labels=n_labels,
                crypto_type=crypto_type,
                crypto_system=crypto_system,
                messengers=messengers,
                compress=compress,
                max_depth=max_depth,
                reg_lambda=reg_lambda,
                min_split_samples=min_split_samples,
                min_split_gain=min_split_gain,
                fix_point_precision=fix_point_precision,
                sampling_method=sampling_method,
                pool=self.pool,
            )
            for _ in range(n_trees)
        ]

    @staticmethod
    def _check_parameters(task, n_labels, compress, sampling_method):
        assert task in ("binary", "multi"), "task should be binary or multi"
        assert n_labels >= 2, "n_labels should be at least 2"
        assert sampling_method in ("uniform",), "sampling method should be uniform"

        if task == "binary":
            assert task == "binary" and n_labels == 2, "binary task should set n_labels as 2"

        if task == "multi":
            if compress is True:
                warnings.warn("compress should be set only when task is binary", SyntaxWarning)

    def train(self, trainset, testset):
        assert isinstance(trainset, NumpyDataset), "trainset should be an instance of NumpyDataset"
        assert isinstance(testset, NumpyDataset), "testset should be an instance of NumpyDataset"

        start_time = time.time()

        print("Building hist...")
        bin_index, bin_split = get_bin_info(trainset.features, self.max_bin)
        print("Done")

        labels = trainset.labels

        if self.task == "binary":
            raw_outputs = np.zeros(len(labels))  # sum of tree raw outputs
            outputs = sigmoid(raw_outputs)  # sigmoid of raw_outputs

            for i, tree in enumerate(self.trees):
                print(f"\ntree {i} started...")
                gradient = self.loss.gradient(labels, outputs)
                hessian = self.loss.hessian(labels, outputs)
                update_pred = tree.fit(gradient, hessian, bin_index, bin_split)
                print(f"tree {i} finished")

                raw_outputs += self.learning_rate * update_pred
                outputs = sigmoid(raw_outputs)

        elif self.task == "multi":
            labels_onehot = np.zeros((len(labels), self.n_labels))
            labels_onehot[np.arange(len(labels)), labels] = 1

            raw_outputs = np.zeros((len(labels), self.n_labels))  # sum of tree raw outputs
            outputs = softmax(raw_outputs, axis=1)  # softmax of raw_outputs

            for i, tree in enumerate(self.trees):
                print(f"\ntree {i} started...")
                gradient = self.loss.gradient(labels_onehot, outputs)
                hessian = self.loss.hessian(labels_onehot, outputs)
                update_pred = tree.fit(gradient, hessian, bin_index, bin_split)
                print(f"tree {i} finished")

                raw_outputs += self.learning_rate * update_pred
                outputs = softmax(raw_outputs, axis=1)

        print("train finished")
        for messenger in self.messengers:
            messenger.send(wrap_message("train finished", content=True))

        if self.pool is not None:
            self.pool.close()

        scores = self._validate(testset)
        print(scores)

        print(colored("Total training and validation time: {:.4f}".format(time.time() - start_time), "red"))

    def _validate(self, testset):
        assert isinstance(testset, NumpyDataset), "testset should be an instance of NumpyDataset"

        features = testset.features
        labels = testset.labels

        raw_outputs = np.zeros((len(labels), self.n_labels)) if self.task == "multi" else np.zeros(len(labels))
        for tree in self.trees:
            update_pred = tree.predict(features)
            raw_outputs += self.learning_rate * update_pred

        if self.task == "binary":
            outputs = sigmoid(raw_outputs)
            targets = np.round(outputs).astype(int)
        elif self.task == "multi":
            outputs = softmax(raw_outputs, axis=1)
            targets = np.argmax(outputs, axis=1)
        else:
            raise ValueError("No such task label.")

        scores = dict()
        acc = accuracy_score(labels, targets)
        scores["acc"] = acc
        if self.task == "binary":
            auc = roc_auc_score(labels, outputs)
            scores["auc"] = auc
            f1 = f1_score(labels, targets, average="weighted")
            scores["f1"] = f1

        print("validate finished")
        for messenger in self.messengers:
            messenger.send(wrap_message("validate finished", content=True))

        return scores

    def predict(self, testset):
        return self._validate(testset)


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "credit"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    n_trees = 5
    task = "binary"
    n_labels = 2
    _crypto_type = Const.PAILLIER
    _key_size = 1024

    compress = True

    active_ips = ["localhost", "localhost"]
    active_ports = [20001, 20002]
    passive_ips = ["localhost", "localhost"]
    passive_ports = [30001, 30002]

    # 1. Load datasets
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(
        dataset_name=dataset_name,
        train=True,
        role=Const.ACTIVE_NAME,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_testset = NumpyDataset.buildin_dataset(
        dataset_name=dataset_name,
        train=False,
        role=Const.ACTIVE_NAME,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_trainset = parse_label(active_trainset)
    active_testset = parse_label(active_testset)
    print("Done")

    # 2. Initialize crypto_system
    crypto_system = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=10000,
        gen_from_set=False,
    )

    # 3. Initialize messenger
    messengers = [messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.ACTIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    ) for active_ip, active_port, passive_ip, passive_port in zip(active_ips, active_ports, passive_ips, passive_ports)]

    # 4. Initialize active tree party and start training
    active_party = ActiveTreeParty(
        n_trees=n_trees,
        task=task,
        n_labels=n_labels,
        crypto_type=_crypto_type,
        crypto_system=crypto_system,
        messengers=messengers,
        compress=compress,
    )
    active_party.train(active_trainset, active_testset)

    # 5. Close messenger, finish training
    for messenger in messengers:
        messenger.close()
