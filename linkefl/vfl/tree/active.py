import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool
from typing import List
from scipy.special import softmax
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
from linkefl.crypto.base import CryptoSystem
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import parse_label
from linkefl.messenger.base import Messenger
from linkefl.modelio import NumpyModelIO
from linkefl.util import sigmoid
from linkefl.vfl.tree import DecisionTree
from linkefl.vfl.tree.data_functions import get_bin_info, wrap_message
from linkefl.vfl.tree.loss_functions import CrossEntropyLoss, MultiCrossEntropyLoss
from linkefl.vfl.tree.plotting import plot_importance


class ActiveTreeParty:
    def __init__(
        self,
        n_trees: int,
        task: str,
        n_labels: int,
        crypto_type: str,
        crypto_system: CryptoSystem,
        messengers: List[Messenger],
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
        saving_model: bool = False,
        model_path: str = "./models",
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
            fix_point_precision: binary length to keep when casting float to int
            sampling_method: uniform
            n_processes: number of processes in multiprocessing
        """

        self._check_parameters(task, n_labels, compress, sampling_method)

        self.n_trees = n_trees
        self.task = task
        self.n_labels = n_labels
        self.messengers = messengers

        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self.saving_model = saving_model
        self.model_path = model_path

        if n_processes > 1:
            self.pool = Pool(n_processes)
        else:
            self.pool = None

        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.ACTIVE_NAME,
            model_type=Const.VERTICAL_SBT,
        )

        self.logger = logger_factory(Const.ACTIVE_NAME)

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
                logger=self.logger,
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

        self.feature_importance_info = {
            "split": defaultdict(int),          # Total number of splits
            "gain": defaultdict(float),         # Total revenue
            "cover": defaultdict(float)           # Total sample covered
        }

    def _check_parameters(self, task, n_labels, compress, sampling_method):
        assert task in ("binary", "multi"), "task should be binary or multi"
        assert n_labels >= 2, "n_labels should be at least 2"
        assert sampling_method in ("uniform",), "sampling should be uniform"

        if task == "binary":
            assert (
                task == "binary" and n_labels == 2
            ), "binary task should set n_labels as 2"

        if task == "multi":
            if compress is True:
                self.logger.log(
                    "compress should be set only when task is binary",
                    level=Const.WARNING,
                )

    def train(self, trainset, testset):
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        start_time = time.time()

        self.logger.log("Building hist...")
        bin_index, bin_split = get_bin_info(trainset.features, self.max_bin)
        self.logger.log("Done")

        labels = trainset.labels

        if self.task == "binary":
            raw_outputs = np.zeros(len(labels))  # sum of tree raw outputs
            outputs = sigmoid(raw_outputs)  # sigmoid of raw_outputs

            for i, tree in enumerate(self.trees):
                self.logger.log(f"tree {i} started...")
                loss = self.loss.loss(labels, outputs)
                gradient = self.loss.gradient(labels, outputs)
                hessian = self.loss.hessian(labels, outputs)
                update_pred = tree.fit(gradient, hessian, bin_index, bin_split, self.feature_importance_info)
                self.logger.log(f"tree {i} finished")

                for messenger in self.messengers:
                    messenger.send(wrap_message("validate", content=True))

                scores = self._validate(testset)
                self.logger.log_metric(
                    epoch=i,
                    loss=loss.mean(),
                    acc=scores["acc"],
                    auc=scores["auc"],
                    f1=scores["f1"],
                    total_epoch=self.n_trees,
                )

                raw_outputs += self.learning_rate * update_pred
                outputs = sigmoid(raw_outputs)

        elif self.task == "multi":
            labels_onehot = np.zeros((len(labels), self.n_labels))
            labels_onehot[np.arange(len(labels)), labels] = 1

            raw_outputs = np.zeros((len(labels), self.n_labels))
            outputs = softmax(raw_outputs, axis=1)  # softmax of raw_outputs

            for i, tree in enumerate(self.trees):
                self.logger.log(f"tree {i} started...")
                loss = self.loss.loss(labels_onehot, outputs)
                gradient = self.loss.gradient(labels_onehot, outputs)
                hessian = self.loss.hessian(labels_onehot, outputs)
                update_pred = tree.fit(gradient, hessian, bin_index, bin_split)
                self.logger.log(f"tree {i} finished")

                for messenger in self.messengers:
                    messenger.send(wrap_message("validate", content=True))

                scores = self._validate(testset)
                self.logger.log_metric(
                    epoch=i,
                    loss=loss.mean(),
                    acc=scores["acc"],
                    auc=scores["auc"],
                    f1=scores["f1"],
                    total_epoch=self.n_trees,
                )

                raw_outputs += self.learning_rate * update_pred
                outputs = softmax(raw_outputs, axis=1)

        for messenger in self.messengers:
            messenger.send(wrap_message("train finished", content=True))

        if self.saving_model:
            model_name = f"{self.model_name}-{trainset.n_samples}_samples.model"
            model_params = [(tree.record, tree.root) for tree in self.trees]
            NumpyModelIO.save(model_params, self.model_path, model_name)

        self.logger.log("train finished")

        if self.pool is not None:
            self.pool.close()

        self.logger.log(
            "Total training and validation time: {:.4f}".format(
                time.time() - start_time
            )
        )

    def _validate(self, testset):
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        features = testset.features
        labels = testset.labels

        raw_outputs = (
            np.zeros((len(labels), self.n_labels))
            if self.task == "multi"
            else np.zeros(len(labels))
        )
        for tree in self.trees:
            update_pred = tree.predict(features)
            if update_pred is None:
                # the trees after are not trained
                break
            raw_outputs += self.learning_rate * update_pred

        if self.task == "binary":
            outputs = sigmoid(raw_outputs)
            targets = np.round(outputs).astype(int)

            acc = accuracy_score(labels, targets)
            auc = roc_auc_score(labels, outputs)
            f1 = f1_score(labels, targets, average="weighted")

        elif self.task == "multi":
            outputs = softmax(raw_outputs, axis=1)
            targets = np.argmax(outputs, axis=1)

            acc = accuracy_score(labels, targets)
            auc = -1
            f1 = -1

        else:
            raise ValueError("No such task label.")

        scores = {"acc": acc, "auc": auc, "f1": f1}

        self.logger.log("validate finished")
        for messenger in self.messengers:
            messenger.send(wrap_message("validate finished", content=True))

        return scores

    def predict(self, testset):
        return self._validate(testset)

    def online_inference(self, dataset, model_name, model_path="./models"):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        model_params = NumpyModelIO.load(model_path, model_name)
        for i, (record, root) in enumerate(model_params):
            tree = self.trees[i]
            tree.record = record
            tree.root = root

        return self._validate(dataset)

    def feature_importances_(self, importance_type="split"):
        """
        Args:
            importance_type: choose in ("split", "gain", "cover"), metrics to evaluate the importance of features.

        Returns:
            dict, include features and importance values.
        """
        assert importance_type in ("split", "gain", "cover"), "Not support evaluation way"

        keys = np.array(list(self.feature_importance_info[importance_type].keys()))
        values = np.array(list(self.feature_importance_info[importance_type].values()))

        if importance_type != 'split':      # The "gain" and "cover" indicators are calculated as mean values
            split_nums = np.array(list(self.feature_importance_info['split'].values()))
            split_nums[split_nums==0] = 1   # Avoid division by zero
            values = values / split_nums

        ascend_index = values.argsort()
        features, values = keys[ascend_index[::-1]], values[ascend_index[::-1]]
        result = {
            'features': list(features),
            f'importance_{importance_type}': list(values)
        }

        return result


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "cancer"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    n_trees = 5
    task = "binary"
    n_labels = 2
    _crypto_type = Const.FAST_PAILLIER
    _key_size = 1024

    n_processes = 6

    active_ips = ["localhost", "localhost"]
    active_ports = [20001, 20002]
    passive_ips = ["localhost", "localhost"]
    passive_ports = [30001, 30002]

    # 1. Load datasets
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                   dataset_name=dataset_name,
                                                   root='../data',
                                                   train=True,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    active_testset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                  dataset_name=dataset_name,
                                                  root='../data',
                                                  train=False,
                                                  download=True,
                                                  passive_feat_frac=passive_feat_frac,
                                                  feat_perm_option=feat_perm_option)
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
    messengers = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
        )
        for active_ip, active_port, passive_ip, passive_port in zip(
            active_ips, active_ports, passive_ips, passive_ports
        )
    ]

    # 4. Initialize active tree party and start training
    active_party = ActiveTreeParty(
        n_trees=n_trees,
        task=task,
        n_labels=n_labels,
        crypto_type=_crypto_type,
        crypto_system=crypto_system,
        messengers=messengers,
        saving_model=True,
        n_processes=n_processes,
    )
    active_party.train(active_trainset, active_testset)
    # scores = active_party.online_inference(active_testset, "xxx.model")
    # print(scores)

    # test
    feature_importance_info = pd.DataFrame(active_party.feature_importances_(importance_type='cover'))
    print(feature_importance_info)

    # 5. Close messenger, finish training
    for messenger in messengers:
        messenger.close()

    ax = plot_importance(active_party, importance_type='split')
    plt.show()
