import datetime
import os
import pathlib
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from linkefl.base import BaseCryptoSystem, BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.messenger.socket_disconnection import FastSocket_disconnection_v1
from linkefl.modelio import NumpyModelIO
from linkefl.util import sigmoid
from linkefl.vfl.tree.data_functions import get_bin_info, wrap_message
from linkefl.vfl.tree.decisiontree import DecisionTree, _DecisionNode
from linkefl.vfl.tree.error import DisconnectedError
from linkefl.vfl.tree.loss_functions import (
    CrossEntropyLoss,
    MeanSquaredErrorLoss,
    MultiCrossEntropyLoss,
)
from linkefl.vfl.utils.evaluate import Evaluate, Plot, TreePrint


class ActiveTreeParty(BaseModelComponent):
    def __init__(
        self,
        n_trees: int,
        task: str,
        n_labels: int,
        crypto_type: str,
        crypto_system: BaseCryptoSystem,
        messengers: List[BaseMessenger],
        logger: GlobalLogger,
        *,
        training_mode: str = "lightgbm",
        learning_rate: float = 0.3,
        compress: bool = False,
        max_bin: int = 16,
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
        n_processes: int = 1,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        drop_protection: bool = False,
        reconnect_ports: List[int] = None,
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

        self._check_parameters(
            training_mode,
            task,
            n_labels,
            compress,
            sampling_method,
            messengers,
            drop_protection,
        )

        self.n_trees = n_trees
        self.task = task
        self.n_labels = n_labels
        self.crypto_type = crypto_type
        self.crypto_system = crypto_system
        self.messengers = messengers
        self.messengers_validTag = [True for _ in range(len(self.messengers))]
        self.model_phase = "online_inference"
        self.logger = logger

        self.learning_rate = learning_rate
        self.max_bin = max_bin
        if n_processes > 1:
            self.pool = Pool(n_processes)
        else:
            self.pool = None
        self.saving_model = saving_model
        if self.saving_model:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if model_dir is None:
                default_dir = "models"
                model_dir = os.path.join(default_dir, self.create_time)
            if model_name is None:
                model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.ACTIVE_NAME,
                        algo_name=Const.AlgoNames.VFL_SBT,
                    )
                    + ".model"
                )
            self.model_dir = model_dir
            self.model_name = model_name
            self.pics_dir = self.model_dir
            if not os.path.exists(self.model_dir):
                pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # 初始化 loss
        if task == "binary":
            self.loss = CrossEntropyLoss()
        elif task == "multi":
            self.loss = MultiCrossEntropyLoss()
        elif task == "regression":
            self.loss = MeanSquaredErrorLoss()
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
                training_mode=training_mode,
                compress=compress,
                max_depth=max_depth,
                max_num_leaves=max_num_leaves,
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
                reconnect_ports=reconnect_ports,
            )
            for _ in range(n_trees)
        ]

        self.feature_importance_info = {
            "split": defaultdict(int),  # Total number of splits
            "gain": defaultdict(float),  # Total revenue
            "cover": defaultdict(float),  # Total sample covered
        }

    def _check_parameters(
        self,
        training_mode,
        task,
        n_labels,
        compress,
        sampling_method,
        messengers,
        drop_protection,
    ):
        assert training_mode in (
            "lightgbm",
            "xgboost",
        ), "training_mode should be lightgbm or xgboost"
        assert task in (
            "regression",
            "binary",
            "multi",
        ), "task should be regression or binary or multi"
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
                    level="warning",
                )

        if task == "regression":
            assert (
                task == "regression" and n_labels == 2
            ), "regression task should set n_labels as 2"

        if drop_protection is True:
            for messenger in messengers:
                assert isinstance(
                    messenger, FastSocket_disconnection_v1
                ), "current messengers type does not support drop protection."

    def fit(
        self,
        trainset: NumpyDataset,
        validset: NumpyDataset,
        role: str = Const.ACTIVE_NAME,
    ) -> None:
        """set for pipeline func."""
        self.train(trainset, validset)

    def train(self, trainset: NumpyDataset, testset: NumpyDataset) -> None:
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
        bin_index, bin_split = get_bin_info(trainset.features, self.max_bin, self.pool)
        self.logger.log("Done")

        # record data for plot fig
        self.mertics_record = {}

        if self.task == "binary" or self.task == "regression":
            raw_outputs = np.zeros(len(trainset.labels))  # sum of tree raw outputs
            raw_outputs_test = np.zeros(len(testset.labels))

            if self.task == "binary":
                outputs = sigmoid(raw_outputs)  # sigmoid of raw_outputs
                outputs_test = sigmoid(raw_outputs_test)
            else:
                outputs = raw_outputs.copy()
                outputs_test = raw_outputs_test.copy()

            residual_record, train_loss_record, test_loss_record = [], [], []
            (
                train_auc_record,
                test_auc_record,
                train_acc_record,
                test_acc_record,
                f1_record,
            ) = ([], [], [], [], [])
            MAE_record, MSE_record, SSE_record, R2_record = [], [], [], []

            for tree_id, tree in enumerate(self.trees):
                self.logger.log(f"tree {tree_id} started...")

                self.logger.log_component(
                    name=Const.VERTICAL_SBT,
                    status=Const.RUNNING,
                    begin=start_time,
                    end=time.time(),
                    duration=time.time() - start_time,
                    progress=tree_id / len(self.trees),
                )

                train_loss = self.loss.loss(labels, outputs)
                gradient = self.loss.gradient(labels, outputs)
                hessian = self.loss.hessian(labels, outputs)

                test_loss = self.loss.loss(testset.labels, outputs_test)

                while True:
                    try:
                        tree.messengers_validTag = (
                            self.messengers_validTag
                        )  # update messengers tag
                        tree.fit(gradient, hessian, bin_index, bin_split)
                        self.messengers_validTag = (
                            tree.messengers_validTag
                        )  # update messengers tag

                        raw_outputs += self.learning_rate * tree.update_pred

                        self._merge_tree_info(tree.feature_importance_info)
                        self.logger.log(f"tree {tree_id} finished")

                        for messenger_id, messenger in enumerate(self.messengers):
                            if self.messengers_validTag[messenger_id]:
                                messenger.send(wrap_message("validate", content=True))

                        # validate
                        scores = self._validate_tree(testset, tree, raw_outputs_test)

                        if self.task == "binary":
                            outputs = sigmoid(raw_outputs)  # sigmoid of raw_outputs
                            outputs_test = sigmoid(raw_outputs_test)
                        else:
                            outputs = raw_outputs.copy()
                            outputs_test = raw_outputs_test.copy()

                    except DisconnectedError as e:
                        # Handling of disconnection during prediction stage
                        tree._reconnect_passiveParty(e.disconnect_party_id)
                    else:
                        break

                residual = (testset.labels - outputs_test).mean()
                residual_record.append(residual)
                train_loss_record.append(train_loss.mean())
                test_loss_record.append(test_loss.mean())

                if self.task == "binary":
                    self.logger.log_metric(
                        epoch=tree_id + 1,
                        loss=train_loss.mean(),
                        acc=scores["acc"],
                        auc=scores["auc"],
                        f1=scores["f1"],
                        ks=scores["ks"],
                        ks_threshold=scores["threshold"],
                        total_epoch=self.n_trees,
                    )
                    train_auc_record.append(roc_auc_score(trainset.labels, outputs))
                    test_auc_record.append(scores["auc"])
                    train_acc_record.append(
                        accuracy_score(trainset.labels, np.round(outputs).astype(int))
                    )
                    test_acc_record.append(scores["acc"])
                    f1_record.append(scores["f1"])
                else:
                    self.logger.log_metric(
                        epoch=tree_id + 1,
                        loss=train_loss.mean(),
                        mae=scores["mae"],
                        mse=scores["mse"],
                        sse=scores["sse"],
                        r2=scores["r2"],
                        total_epoch=self.n_trees,
                    )
                    MAE_record.append(scores["mae"])
                    MSE_record.append(scores["mse"])
                    SSE_record.append(scores["sse"])
                    R2_record.append(scores["r2"])

        elif self.task == "multi":
            labels_onehot = np.zeros((len(labels), self.n_labels))
            labels_onehot[np.arange(len(labels)), labels] = 1

            (
                train_acc_record,
                test_acc_record,
            ) = ([], [])

            raw_outputs = np.zeros((len(labels), self.n_labels))
            outputs = softmax(raw_outputs, axis=1)  # softmax of raw_outputs

            raw_outputs_test = np.zeros((len(testset.labels), self.n_labels))

            for tree_id, tree in enumerate(self.trees):
                self.logger.log(f"tree {tree_id} started...")

                self.logger.log_component(
                    name=Const.VERTICAL_SBT,
                    status=Const.RUNNING,
                    begin=start_time,
                    end=time.time(),
                    duration=time.time() - start_time,
                    progress=tree_id / len(self.trees),
                )

                loss = self.loss.loss(labels_onehot, outputs)
                gradient = self.loss.gradient(labels_onehot, outputs)
                hessian = self.loss.hessian(labels_onehot, outputs)

                while True:
                    try:
                        tree.messengers_validTag = (
                            self.messengers_validTag
                        )  # update messengers tag
                        tree.fit(gradient, hessian, bin_index, bin_split)

                        raw_outputs += self.learning_rate * tree.update_pred
                        outputs = softmax(raw_outputs, axis=1)

                        self.messengers_validTag = (
                            tree.messengers_validTag
                        )  # update messengers tag
                        self._merge_tree_info(tree.feature_importance_info)
                        self.logger.log(f"tree {tree_id} finished")

                        for messenger_id, messenger in enumerate(self.messengers):
                            if self.messengers_validTag[messenger_id]:
                                messenger.send(wrap_message("validate", content=True))

                        scores = self._validate_tree(testset, tree, raw_outputs_test)

                    except DisconnectedError as e:
                        # Handling of disconnection during prediction stage
                        tree._reconnect_passiveParty(e.disconnect_party_id)
                    else:
                        break

                self.logger.log_metric(
                    epoch=tree_id + 1,
                    loss=loss.mean(),
                    acc=scores["acc"],
                    auc=scores["auc"],
                    f1=scores["f1"],
                    total_epoch=self.n_trees,
                )

        for messenger_id, messenger in enumerate(self.messengers):
            if self.messengers_validTag[messenger_id]:
                messenger.send(wrap_message("train finished", content=True))

        self.logger.log_component(
            name=Const.VERTICAL_SBT,
            status=Const.SUCCESS,
            begin=start_time,
            end=time.time(),
            duration=time.time() - start_time,
            progress=1.0,
        )
        self.logger.log("train finished")
        self.logger.log(
            "Total training and validation time: {:.4f}".format(
                time.time() - start_time
            )
        )

        if self.pool is not None:
            self.pool.close()

        if self.saving_model:  # save training files.
            self._save_model()
            model_structure = self.get_tree_str_structures()
            for messenger_id, messenger in enumerate(self.messengers):
                if self.messengers_validTag[messenger_id]:
                    messenger.send(model_structure)

            # 输出模型
            tree_strs = self.get_tree_str_structures(tree_structure="VERTICAL")
            Plot.plot_trees(tree_strs, self.pics_dir)
            Plot.plot_importance(self, importance_type="split", file_dir=self.pics_dir)

            if self.task == "regression":
                Plot.plot_train_test_loss(
                    train_loss_record, test_loss_record, self.pics_dir
                )
                Plot.plot_residual(residual_record, self.pics_dir)
                Plot.plot_ordered_lorenz_curve(
                    label=testset.labels, y_prob=outputs_test, file_dir=self.pics_dir
                )
                Plot.plot_predict_distribution(
                    y_prob=outputs, bins=10, file_dir=self.pics_dir
                )
                Plot.plot_predict_prob_box(y_prob=outputs, file_dir=self.pics_dir)
                Plot.plot_regression_metrics(
                    MAE_record, MSE_record, SSE_record, R2_record, self.pics_dir
                )
            elif self.task == "binary":
                Plot.plot_train_test_loss(
                    train_loss_record, test_loss_record, self.pics_dir
                )
                Plot.plot_residual(residual_record, self.pics_dir)
                Plot.plot_train_test_auc(
                    train_auc_record, test_auc_record, self.pics_dir
                )
                Plot.plot_train_test_acc(
                    train_acc_record, test_acc_record, self.pics_dir
                )
                Plot.plot_ordered_lorenz_curve(
                    label=testset.labels, y_prob=outputs_test, file_dir=self.pics_dir
                )
                Plot.plot_predict_distribution(
                    y_prob=outputs, bins=10, file_dir=self.pics_dir
                )
                Plot.plot_predict_prob_box(y_prob=outputs, file_dir=self.pics_dir)
                Plot.plot_binary_mertics(
                    testset.labels, outputs_test, cut_point=50, file_dir=self.pics_dir
                )
                Plot.plot_f1_score(f1_record, self.pics_dir)
            else:
                pass


    def score(
        self, testset: NumpyDataset, role: str = Const.ACTIVE_NAME
    ) -> Dict[str, float]:
        """set for pipeline func."""
        return self._validate(testset)

    @staticmethod
    def online_inference(
        dataset: NumpyDataset,
        messengers: List[BaseMessenger],
        logger: GlobalLogger,
        model_dir: str,
        model_name: str,
        role: str = Const.ACTIVE_NAME,
    ):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        (
            model_params,
            task,
            n_labels,
            learning_rate,
            feature_importance_info,
            tree_structures,
        ) = NumpyModelIO.load(model_dir, model_name)

        model = ActiveTreeParty(
            n_trees=len(model_params),
            task=task,
            n_labels=n_labels,
            crypto_type="",
            crypto_system=None,
            messengers=messengers,
            logger=logger,
        )

        model.trees = [
            DecisionTree(
                task=task,
                n_labels=n_labels,
                crypto_type=None,
                crypto_system=None,
                messengers=messengers,
                logger=logger,
            )
            for _ in range(len(model_params))
        ]

        for i, (record, root) in enumerate(model_params):
            model.trees[i].record = record
            model.trees[i].root = root

        preds = model._pred(dataset)
        for messenger in messengers:
            messenger.send(preds)
        return preds

    def _pred(self, testset):
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

        if self.task == "regression":
            outputs = raw_outputs
        elif self.task == "binary":
            outputs = sigmoid(raw_outputs)
        elif self.task == "multi":
            outputs = softmax(raw_outputs, axis=1)
        else:
            raise ValueError("No such task label.")

        return outputs


    def feature_importances_(self, importance_type: str = "split") -> Dict[str, list]:
        """
        Args:
            importance_type: choose in ("split", "gain", "cover"),
            metrics to evaluate the importance of features.

        Returns:
            dict, include features and importance values.
        """
        assert importance_type in (
            "split",
            "gain",
            "cover",
        ), "Not support evaluation way"

        keys = np.array(list(self.feature_importance_info[importance_type].keys()))
        values = np.array(list(self.feature_importance_info[importance_type].values()))

        if (
            importance_type != "split"
        ):  # The "gain" and "cover" indicators are calculated as mean values
            split_nums = np.array(list(self.feature_importance_info["split"].values()))
            split_nums[split_nums == 0] = 1  # Avoid division by zero
            values = values / split_nums

        ascend_index = values.argsort()
        features, values = keys[ascend_index[::-1]], values[ascend_index[::-1]]
        result = {
            "features": list(features),
            f"importance_{importance_type}": list(values),
        }

        return result

    def load_model(self, model_name: str, model_path: str = "./models") -> None:
        (
            model_params,
            task,
            n_labels,
            learning_rate,
            feature_importance_info,
            tree_structures,
        ) = NumpyModelIO.load(model_path, model_name)

        self.task = (task,)
        self.n_labels = (n_labels,)
        self.learning_rate = (learning_rate,)
        self.trees = [
            DecisionTree(
                task=task,
                n_labels=n_labels,
                crypto_type=None,
                crypto_system=None,
                messengers=messengers,
                logger=logger,
            )
            for _ in range(len(model_params))
        ]

        for i, (record, root) in enumerate(model_params):
            self.trees[i].record = record
            self.trees[i].root = root

        self.feature_importance_info = feature_importance_info
        self.logger.log(f"Load model {model_name} success.")

    def get_tree_str_structures(
        self, tree_structure: str = "VERTICAL"
    ) -> Dict[str, str]:
        assert tree_structure in [
            "VERTICAL",
            "HORIZONTAL",
        ], "tree_structure should be VERTICAL or HORIZONTAL"
        tree_strs = {}

        for tree_id, tree in enumerate(self.trees, 1):
            tree_str = TreePrint.tree_to_str(tree, tree_structure)
            # tree_str = tree_to_str(tree, tree_structure)
            tree_strs[f"tree{tree_id}"] = tree_str

        self.logger.log(f"Load model {self.model_name} success.")
        return tree_strs

    def get_tree_structures(self) -> Dict[str, str]:
        """tree_id : 1-based"""

        def _pre_order_traverse(root):
            nonlocal data

            if not root:
                data += "None;"
                return

            node_info = ""

            if root.value is not None:
                # leaf node
                node_info += f"leaf value: {root.value: .4f},"
            else:
                # mid node
                if root.party_id == 0:
                    node_info += "active_party,"
                    node_info += f"record_id: {root.record_id},"
                    node_info += f"split_feature: f{tree.record[root.record_id][0]},"
                    node_info += f"split_value: {tree.record[root.record_id][1]: .4f},"
                else:
                    node_info += f"passive_party_{root.party_id},"
                    node_info += f"record_id: {root.record_id},"
                    node_info += "split_feature: encrypt,"
                    node_info += "split_value: encrypt,"

            data += f"{node_info};"

            if root.left_branch:
                _pre_order_traverse(root.left_branch)
            if root.right_branch:
                _pre_order_traverse(root.right_branch)

        tree_structures = {}

        for tree_id, tree in enumerate(self.trees, 1):
            data = ""
            _pre_order_traverse(tree.root)
            tree_structures[f"tree{tree_id}"] = data

        return tree_structures

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

        if self.task == "regression":
            outputs = raw_outputs_test
            targets = raw_outputs_test

            mae = mean_absolute_error(labels, outputs)
            mse = mean_squared_error(labels, outputs)
            sse = mse * len(labels)
            r2 = r2_score(labels, outputs)
            scores = {"mae": mae, "mse": mse, "sse": sse, "r2": r2}

        elif self.task == "binary":
            outputs = sigmoid(raw_outputs_test)
            targets = np.round(outputs).astype(int)

            acc = accuracy_score(labels, targets)
            auc = roc_auc_score(labels, outputs)
            f1 = f1_score(labels, targets, average="weighted")
            ks_value, threshold = Evaluate.eval_ks(labels, targets, cut_point=50)
            scores = {
                "acc": acc,
                "auc": auc,
                "f1": f1,
                "ks": ks_value,
                "threshold": threshold,
            }

        elif self.task == "multi":
            outputs = softmax(raw_outputs_test, axis=1)
            targets = np.argmax(outputs, axis=1)

            acc = accuracy_score(labels, targets)
            auc = -1
            f1 = -1
            scores = {"acc": acc, "auc": auc, "f1": f1}

        else:
            raise ValueError("No such task label.")

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

        if self.task == "regression":
            outputs = raw_outputs
            targets = raw_outputs

            mae = mean_absolute_error(labels, outputs)
            mse = mean_squared_error(labels, outputs)
            sse = mse * len(labels)
            r2 = r2_score(labels, outputs)
            scores = {"mae": mae, "mse": mse, "sse": sse, "r2": r2, "targets": targets}

        elif self.task == "binary":
            outputs = sigmoid(raw_outputs)
            targets = np.round(outputs).astype(int)

            acc = accuracy_score(labels, targets)
            auc = roc_auc_score(labels, outputs)
            f1 = f1_score(labels, targets, average="weighted")
            ks_value, threshold = Evaluate.eval_ks(labels, targets, cut_point=50)
            scores = {
                "acc": acc,
                "auc": auc,
                "f1": f1,
                "ks": ks_value,
                "threshold": threshold,
                "targets": targets,
            }

        elif self.task == "multi":
            outputs = softmax(raw_outputs, axis=1)
            targets = np.argmax(outputs, axis=1)

            acc = accuracy_score(labels, targets)
            auc = -1
            f1 = -1
            scores = {"acc": acc, "auc": auc, "f1": f1, "targets": targets}

        else:
            raise ValueError("No such task label.")

        for i, messenger in enumerate(self.messengers):
            if self.messengers_validTag[i]:
                messenger.send(wrap_message("validate finished", content=True))

        self.logger.log("validate finished")

        return scores

    def _merge_tree_info(self, feature_importance_info_tree):
        if feature_importance_info_tree is not None:
            for key in feature_importance_info_tree["split"].keys():
                self.feature_importance_info["split"][
                    key
                ] += feature_importance_info_tree["split"][key]
            for key in feature_importance_info_tree["gain"].keys():
                self.feature_importance_info["gain"][
                    key
                ] += feature_importance_info_tree["gain"][key]
            for key in feature_importance_info_tree["cover"].keys():
                self.feature_importance_info["cover"][
                    key
                ] += feature_importance_info_tree["cover"][key]

        self.logger.log("merge tree information done")

    def _save_model(self):
        # model_name = f"{self.model_name}.model"
        model_name = self.model_name

        for tree in self.trees:
            self._removing_useless_message(tree.root)

        model_params = [(tree.record, tree.root) for tree in self.trees]
        model_structures = self.get_tree_str_structures()
        saved_data = [
            model_params,
            self.task,
            self.n_labels,
            self.learning_rate,
            self.feature_importance_info,
            model_structures,
        ]
        NumpyModelIO.save(saved_data, self.model_dir, model_name)
        self.logger.log(f"Save model {model_name} success.")

    def _removing_useless_message(self, root: _DecisionNode):
        if not root:
            return

        del (
            root.hist_list,
            root.sample_tag_selected,
            root.sample_tag_unselected,
            root.split_party_id,
            root.split_gain,
            root.split_bin_id,
            root.split_feature_id,
            root.depth,
        )

        if root.left_branch:
            self._removing_useless_message(root.left_branch)
        if root.right_branch:
            self._removing_useless_message(root.right_branch)


if __name__ == "__main__":
    pass

    from linkefl.common.factory import (
        crypto_factory,
        logger_factory,
        messenger_factory,
        messenger_factory_disconnection,
    )
    from linkefl.feature.transform import parse_label

    # 0. Set parameters
    #  binary: cancer, digits, epsilon, census, credit, default_credit, criteo
    #  regression: diabetes
    dataset_name = "cancer"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    n_trees = 5
    task = "binary"  # multi, binary, regression
    n_labels = 2
    _crypto_type = Const.FAST_PAILLIER
    _key_size = 1024

    n_processes = 6

    active_ips = [
        "localhost",
    ]
    active_ports = [
        21001,
    ]
    passive_ips = [
        "localhost",
    ]
    passive_ports = [
        20001,
    ]

    drop_protection = False
    reconnect_ports = [30003]

    # 1. Load datasets
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_testset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root="../data",
        train=False,
        download=True,
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

    # 3. Initialize messengers
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
                model_type="Tree",  # used as tag to verify data
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
    logger = logger_factory(role=Const.ACTIVE_NAME)
    active_party = ActiveTreeParty(
        n_trees=n_trees,
        task=task,
        n_labels=n_labels,
        crypto_type=_crypto_type,
        crypto_system=crypto_system,
        messengers=messengers,
        logger=logger,
        training_mode="lightgbm",  # "lightgbm", "xgboost"
        sampling_method="uniform",
        max_depth=6,
        max_num_leaves=8,
        subsample=1,
        top_rate=0.3,
        other_rate=0.7,
        colsample_bytree=1,
        n_processes=n_processes,
        drop_protection=drop_protection,
        reconnect_ports=reconnect_ports,
        saving_model=True,
        # model_dir="./models"
    )

    # active_party.train(active_trainset, active_testset)
    scores, targets = active_party.online_inference(
        active_testset,
        messengers,
        logger,
        model_dir="./models/20230404161927",
        model_name="20230404161927-active_party-vfl_sbt.model",
    )
    print(scores, targets)
    # feature_importance_info = pd.DataFrame(
    #     active_party.feature_importances_(importance_type='cover')
    # )
    # print(feature_importance_info)

    # ax = plot_importance(active_party, importance_type='split')
    # plt.show()

    # trees_strs = active_party.plot_trees(tree_structure="VERTICAL")
    # print(trees_strs[1])        # tree id is 1-based

    # 5. Close messengers, finish training
    for messenger in messengers:
        messenger.close()
