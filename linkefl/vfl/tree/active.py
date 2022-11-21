import datetime
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from multiprocessing import Pool
from typing import List
from scipy.special import softmax
from collections import defaultdict
from termcolor import colored
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory, messenger_factory_disconnection
from linkefl.crypto.base import CryptoSystem
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import parse_label
from linkefl.messenger.base import Messenger
from linkefl.messenger.socket_disconnection import FastSocket_disconnection_v1
from linkefl.modelio import NumpyModelIO
from linkefl.pipeline.base import ModelComponent
from linkefl.util import sigmoid
from linkefl.vfl.tree import DecisionTree
from linkefl.vfl.tree.error import DisconnectedError
from linkefl.vfl.tree.data_functions import get_bin_info, wrap_message
from linkefl.vfl.tree.loss_functions import CrossEntropyLoss, MultiCrossEntropyLoss
from linkefl.vfl.tree.plotting import plot_importance


class ActiveTreeParty(ModelComponent):
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
        subsample: float = 1,
        top_rate: float = 0.5,
        other_rate: float = 0.5,
        colsample_bytree: float = 1,
        n_processes: int = 1,
        saving_model: bool = False,
        model_path: str = "./models",
        drop_protection: bool = False,
        reconnect_ports: list = []
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
            sampling_method: uniform or goss
            subsample: sample sampling ratio for uniform sampling
            top_rate: head sample retention ratio for goss sampling
            other_rate: tail sample sampling ratio for goss sampling
            colsample_bytree: tree-level feature sampling scale
            n_processes: number of processes in multiprocessing
        """

        self._check_parameters(task, n_labels, compress, sampling_method, messengers, drop_protection)

        self.n_trees = n_trees
        self.task = task
        self.n_labels = n_labels
        self.messengers = messengers
        self.messengers_validTag = [True for _ in range(len(self.messengers))]
        self.model_phase = "online_inference"

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
                subsample=subsample,
                top_rate=top_rate,
                other_rate=other_rate,
                colsample_bytree=colsample_bytree,
                pool=self.pool,
                drop_protection=drop_protection,
                reconnect_ports=reconnect_ports
            )
            for _ in range(n_trees)
        ]

        self.feature_importance_info = {
            "split": defaultdict(int),          # Total number of splits
            "gain": defaultdict(float),         # Total revenue
            "cover": defaultdict(float)         # Total sample covered
        }

    def _check_parameters(self, task, n_labels, compress, sampling_method, messengers, drop_protection):
        assert task in ("binary", "multi"), "task should be binary or multi"
        assert n_labels >= 2, "n_labels should be at least 2"
        assert sampling_method in ("uniform", "goss"), "sampling method not supported"

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

        if drop_protection is True:
            for messenger in messengers:
                assert isinstance(
                    messenger, FastSocket_disconnection_v1
                ), "current messenger type does not support drop protection."

    def fit(self, trainset, testset, role=Const.ACTIVE_NAME):
        """set for pipeline func.
        """
        self.train(trainset, testset)

    def train(self, trainset, testset):
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        start_time = time.time()

        self.model_phase = "train"
        labels = trainset.labels

        self.logger.log("Building hist...")
        bin_index, bin_split = get_bin_info(trainset.features, self.max_bin)
        self.logger.log("Done")

        if self.task == "binary":
            raw_outputs = np.zeros(len(labels))  # sum of tree raw outputs
            outputs = sigmoid(raw_outputs)  # sigmoid of raw_outputs

            raw_outputs_test = np.zeros(len(testset.labels))

            for tree_id, tree in enumerate(self.trees):
                self.logger.log(f"tree {tree_id} started...")

                loss = self.loss.loss(labels, outputs)
                gradient = self.loss.gradient(labels, outputs)
                hessian = self.loss.hessian(labels, outputs)

                while True:
                    try:
                        tree.messengers_validTag = self.messengers_validTag         # update messengers tag
                        fit_result = tree.fit(gradient, hessian, bin_index, bin_split)
                        self.messengers_validTag = tree.messengers_validTag         # update messengers tag

                        raw_outputs += self.learning_rate * fit_result["update_pred"]
                        outputs = sigmoid(raw_outputs)

                        self._merge_tree_info(fit_result["feature_importance_info"])
                        self.logger.log(f"tree {i} finished")

                        for messenger_id, messenger in enumerate(self.messengers):
                            if self.messengers_validTag[messenger_id]:
                                messenger.send(wrap_message("validate", content=True))
                        # scores = self._validate(testset)
                        scores = self._validate_tree(testset, tree, raw_outputs_test)
                    except DisconnectedError as e:
                        # Handling of disconnection during prediction stage
                        tree._reconnect_passiveParty(e.disconnect_party_id)
                    else:
                        break

                self.logger.log_metric(
                    epoch=i,
                    loss=loss.mean(),
                    acc=scores["acc"],
                    auc=scores["auc"],
                    f1=scores["f1"],
                    total_epoch=self.n_trees,
                )

        elif self.task == "multi":
            labels_onehot = np.zeros((len(labels), self.n_labels))
            labels_onehot[np.arange(len(labels)), labels] = 1

            raw_outputs = np.zeros((len(labels), self.n_labels))
            outputs = softmax(raw_outputs, axis=1)  # softmax of raw_outputs

            raw_outputs_test = np.zeros((len(testset.labels), self.n_labels))

            for tree_id, tree in enumerate(self.trees):
                self.logger.log(f"tree {tree_id} started...")

                loss = self.loss.loss(labels_onehot, outputs)
                gradient = self.loss.gradient(labels_onehot, outputs)
                hessian = self.loss.hessian(labels_onehot, outputs)

                while True:
                    try:
                        tree.messengers_validTag = self.messengers_validTag  # update messengers tag
                        fit_result = tree.fit(gradient, hessian, bin_index, bin_split)

                        raw_outputs += self.learning_rate * fit_result["update_pred"]
                        outputs = softmax(raw_outputs, axis=1)

                        self.messengers_validTag = tree.messengers_validTag  # update messengers tag
                        self._merge_tree_info(fit_result["feature_importance_info"])
                        self.logger.log(f"tree {i} finished")

                        for messenger_id, messenger in enumerate(self.messengers):
                            if self.messengers_validTag[messenger_id]:
                                messenger.send(wrap_message("validate", content=True))

                        # scores = self._validate(testset)
                        scores = self._validate_tree(testset, tree, raw_outputs_test)
                    except DisconnectedError as e:
                        # Handling of disconnection during prediction stage
                        tree._reconnect_passiveParty(e.disconnect_party_id)
                    else:
                        break

                self.logger.log_metric(
                    epoch=i,
                    loss=loss.mean(),
                    acc=scores["acc"],
                    auc=scores["auc"],
                    f1=scores["f1"],
                    total_epoch=self.n_trees,
                )

        for messenger_id, messenger in enumerate(self.messengers):
            if self.messengers_validTag[messenger_id]:
                messenger.send(wrap_message("train finished", content=True))

        self.logger.log("train finished")
        self.logger.log("Total training and validation time: {:.4f}".format(time.time() - start_time))

        if self.pool is not None:
            self.pool.close()

        if self.saving_model:
            self._save_model()

    def score(self, testset, role=Const.ACTIVE_NAME):
        """set for pipeline func.
        """
        return self._validate(testset)

    def online_inference(self, dataset, model_name, model_path="./models"):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"

        self.load_model(model_name, model_path)

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

    def load_model(self, model_name, model_path="./models"):
        model_params, feature_importance_info = NumpyModelIO.load(model_path, model_name)

        if len(self.trees) != len(model_params):
            self.trees = [
                DecisionTree(
                    task=self.task,
                    n_labels=self.n_labels,
                    crypto_type=self.crypto_type,
                    crypto_system=self.crypto_system,
                    messengers=self.messengers,
                    logger=self.logger
                )
                for _ in range(len(model_params))
            ]

        for i, (record, root) in enumerate(model_params):
            tree = self.trees[i]
            tree.record = record
            tree.root = root

        self.feature_importance_info = feature_importance_info
        self.logger.log(f"Load model {model_name} success.")

    def _validate_tree(self, testset, tree, raw_outputs_test=None):
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        features = testset.features
        labels = testset.labels

        if raw_outputs_test is None:
            if self.task == "multi":
                raw_outputs_test = np.zeros((len(labels), self.n_labels))
            else:
                raw_outputs_test = np.zeros(len(labels))

        update_pred = tree.predict(features)
        raw_outputs_test += self.learning_rate * update_pred

        if self.task == "binary":
            outputs = sigmoid(raw_outputs_test)
            targets = np.round(outputs).astype(int)

            acc = accuracy_score(labels, targets)
            auc = roc_auc_score(labels, outputs)
            f1 = f1_score(labels, targets, average="weighted")

        elif self.task == "multi":
            outputs = softmax(raw_outputs_test, axis=1)
            targets = np.argmax(outputs, axis=1)

            acc = accuracy_score(labels, targets)
            auc = -1
            f1 = -1

        else:
            raise ValueError("No such task label.")

        scores = {"acc": acc, "auc": auc, "f1": f1}

        for i, messenger in enumerate(self.messengers):
            if self.messengers_validTag[i]:
                messenger.send(wrap_message("validate finished", content=True))

        self.logger.log("validate finished")

        # TODO: test wheather need to return raw_outputs
        # return raw_outputs_test, scores
        return scores

    def _validate(self, testset):
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        features = testset.features
        labels = testset.labels

        if self.task == "multi":
            raw_outputs = np.zeros((len(labels), self.n_labels))
        else:
            raw_outputs = np.zeros(len(labels))

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

        for i, messenger in enumerate(self.messengers):
            if self.messengers_validTag[i]:
                messenger.send(wrap_message("validate finished", content=True))

        self.logger.log("validate finished")

        return scores

    def _merge_tree_info(self, feature_importance_info_tree):
        if feature_importance_info_tree is not None:
            for key in feature_importance_info_tree["split"].keys():
                self.feature_importance_info["split"][key] += feature_importance_info_tree["split"][key]
            for key in feature_importance_info_tree["gain"].keys():
                self.feature_importance_info["gain"][key] += feature_importance_info_tree["gain"][key]
            for key in feature_importance_info_tree["cover"].keys():
                self.feature_importance_info["cover"][key] += feature_importance_info_tree["cover"][key]

        self.logger.log("merge tree information done")

    def _save_model(self):
        model_name = f"{self.model_name}.model"
        model_params = [(tree.record, tree.root) for tree in self.trees]
        saved_data = [model_params, self.feature_importance_info]
        NumpyModelIO.save(saved_data, self.model_path, model_name)

        self.logger.log(f"Save model {model_name} success.")

if __name__ == "__main__":
    # 0. Set parameters
    # cancer, digits, epsilon, census, credit, default_credit, criteo
    dataset_name = "cancer"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    n_trees = 5
    task = "binary"     # multi, binary
    n_labels = 2
    _crypto_type = Const.FAST_PAILLIER
    _key_size = 1024

    n_processes = 6

    active_ips = ["localhost", "localhost"]
    active_ports = [20001, 20002]
    passive_ips = ["localhost", "localhost"]
    passive_ports = [30001, 30002]

    drop_protection = True
    reconnect_ports = [30003, 30004]

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
    if not drop_protection:
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
    else:
        messengers = [
            messenger_factory_disconnection(
                messenger_type=Const.FAST_SOCKET_V1,
                role=Const.ACTIVE_NAME,
                model_type='Tree',                   # used as tag to verify data
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
        sampling_method='goss',
        subsample=0.9,
        top_rate=0.3,
        other_rate=0.7,
        colsample_bytree=1,
        saving_model=True,
        n_processes=n_processes,

        drop_protection=drop_protection,
        reconnect_ports=reconnect_ports
    )

    active_party.train(active_trainset, active_testset)

    feature_importance_info = pd.DataFrame(active_party.feature_importances_(importance_type='cover'))
    print(feature_importance_info)

    # ax = plot_importance(active_party, importance_type='split')
    # plt.show()

    # scores = active_party.online_inference(active_testset, "xxx.model")
    # print(scores)

    # 5. Close messenger, finish training
    for messenger in messengers:
        messenger.close()

