import datetime
import os
import pathlib
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np
from scipy.special import softmax

from linkefl.base import BaseCryptoSystem, BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.messenger.socket_disconnection import FastSocket_disconnection_v1
from linkefl.modelio import NumpyModelIO
from linkefl.util import sigmoid
from linkefl.vfl.tree.core.decision_tree import DecisionTree
from linkefl.vfl.tree.core.error import DisconnectedError
from linkefl.vfl.tree.core.loss_funcs import (
    CrossEntropyLoss,
    MeanSquaredErrorLoss,
    MultiCrossEntropyLoss,
)
from linkefl.vfl.tree.core.tree_node import _DecisionNode
from linkefl.vfl.tree.utils.bins import Bins
from linkefl.vfl.tree.utils.mertics import Metrics
from linkefl.vfl.tree.utils.print_tree import PrintTree
from linkefl.vfl.tree.utils.util_func import wrap_message, one_hot
from linkefl.vfl.utils.evaluate import Plot


class ActiveTreeParty(BaseModelComponent):
    def __init__(
        self,
        n_trees: int,
        task: str,
        n_labels: int,
        crypto_type: str,
        crypto_system: Optional[BaseCryptoSystem],
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
        reg_gamma: float = 0.1,
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
    ) -> None:
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

        if task == "binary":
            self.loss = CrossEntropyLoss()
        elif task == "multi":
            self.loss = MultiCrossEntropyLoss()
        elif task == "regression":

            self.loss = MeanSquaredErrorLoss()
        else:
            raise ValueError("No such task label.")

        compress = True if task == "binary" else False

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
                reg_gamma=reg_gamma,
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

        if task == "regression":
            assert (
                task == "regression" and n_labels == 2
            ), "regression task should set n_labels as 2"

        if task == "binary":
            assert (
                task == "binary" and n_labels == 2
            ), "binary task should set n_labels as 2"

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

    def train(
        self,
        trainset: NumpyDataset,
        testset: NumpyDataset,
    ) -> None:
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        self.logger.log("Training start.")

        bin_index, bin_split = Bins.get_bin_info(
            trainset.features, self.max_bin, self.pool
        )
        self.logger.log("Build histogram information done.")

        if self.task == "regression" or self.task == "binary":
            train_labels = trainset.labels
            test_labels = testset.labels
            raw_outputs = np.zeros(len(trainset.labels))  # sum of tree raw outputs
            raw_outputs_test = np.zeros(len(testset.labels))
            if self.task == "binary":
                outputs = sigmoid(raw_outputs)  # sigmoid of raw_outputs
                outputs_test = sigmoid(raw_outputs_test)
            else:
                outputs = raw_outputs.copy()
                outputs_test = raw_outputs_test.copy()
        else:
            train_labels = one_hot(trainset.labels, self.n_labels)
            test_labels = one_hot(testset.labels, self.n_labels)
            raw_outputs = np.zeros((len(trainset.labels), self.n_labels))
            raw_outputs_test = np.zeros((len(testset.labels), self.n_labels))
            outputs = softmax(raw_outputs, axis=1)  # softmax of raw_outputs
            outputs_test = softmax(raw_outputs_test, axis=1)  # softmax of raw_outputs

        metrics_record = Metrics(self.task)
        for tree_id, tree in enumerate(self.trees):
            self.logger.log(f"Start training tree {tree_id}.")

            train_loss = self.loss.loss(train_labels, outputs)
            test_loss = self.loss.loss(test_labels, outputs_test)
            gradient = self.loss.gradient(train_labels, outputs)
            hessian = self.loss.hessian(train_labels, outputs)

            while True:  # train a tree
                try:
                    tree.messengers_validTag = self.messengers_validTag
                    tree.fit(gradient, hessian, bin_index, bin_split)
                    self.messengers_validTag = tree.messengers_validTag

                    raw_outputs += self.learning_rate * tree.update_pred

                    # validate tree
                    update_pred_test = self._validate_tree(tree, testset)
                    raw_outputs_test += self.learning_rate * update_pred_test

                    if self.task == "regression":
                        outputs = raw_outputs.copy()
                        outputs_test = raw_outputs_test.copy()
                    elif self.task == "binary":
                        outputs = sigmoid(raw_outputs)
                        outputs_test = sigmoid(raw_outputs_test)
                    else:
                        outputs = softmax(raw_outputs, axis=1)
                        outputs_test = softmax(raw_outputs_test, axis=1)

                    scores = Metrics.cal_mertics(
                        task=self.task, labels=testset.labels, y_preds=outputs_test
                    )

                    self._merge_tree_info(tree.feature_importance_info)

                except DisconnectedError as e:
                    # Handling of disconnection during prediction stage
                    tree._reconnect_passiveParty(e.disconnect_party_id)
                else:
                    break

            self.logger.log(f"Tree {tree_id} finished.")
            if self.task == "regression":
                self.logger.log_metric(
                    epoch=tree_id + 1,
                    loss=train_loss.mean(),
                    mae=scores["mae"],
                    mse=scores["mse"],
                    sse=scores["sse"],
                    r2=scores["r2"],
                    total_epoch=self.n_trees,
                )
                metrics_record.record_scores(scores)
                metrics_record.record_loss(train_loss, test_loss)
                metrics_record.record_residual(testset.labels, outputs_test)
            elif self.task == "binary":
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
                metrics_record.record_scores(scores)
                metrics_record.record_loss(train_loss, test_loss)
                metrics_record.record_residual(testset.labels, outputs_test)
                metrics_record.record_train_acc_auc(trainset.labels, outputs)
            elif self.task == "multi":
                self.logger.log_metric(
                    epoch=tree_id + 1,
                    loss=train_loss.mean(),
                    acc= scores["acc"],
                    total_epoch=self.n_trees,
                )
                metrics_record.record_scores(scores)
                metrics_record.record_loss(train_loss, test_loss)
            else:
                raise ValueError("Not such task.")

        # save model and training fig.
        if self.saving_model:
            self.save_model()
            PrintTree.plot_tree_strs(
                trees=self.trees, tree_structure="VERTICAL", file_dir=self.pics_dir
            )
            Plot.plot_importance(self, importance_type="split", file_dir=self.pics_dir)
            metrics_record.save_mertic_pics(
                labels=testset.labels, y_preds=outputs_test, pics_dir=self.pics_dir
            )

        for messenger_id, messenger in enumerate(self.messengers):
            if self.messengers_validTag[messenger_id]:
                messenger.send(wrap_message("train finished", content=True))

        if self.pool is not None:
            self.pool.close()

        self.logger.log("Training finished.")

    def score(
        self,
        testset: NumpyDataset,
        role: str = Const.ACTIVE_NAME,
    ) -> Dict[str, float]:
        """set for pipeline func."""
        preds = self.predict(testset)
        scores = Metrics.cal_mertics(
            task=self.task, labels=testset.labels, y_preds=preds
        )
        return scores

    def predict(
        self,
        testset: NumpyDataset,
    ) -> List[float]:
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        if self.task == "multi":
            raw_outputs = np.zeros((len(testset.labels), self.n_labels))
        else:
            raw_outputs = np.zeros(len(testset.labels))

        self.logger.log("Start to predict.")

        for tree in self.trees:
            update_pred = tree.predict(testset.features)
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
            raise ValueError("Unsupported task.")

        for i, messenger in enumerate(self.messengers):
            if self.messengers_validTag[i]:
                messenger.send(wrap_message("validate finished", content=True))

        self.logger.log("Predict finished.")

        return outputs

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

        model.logger.log(f"Load model {model_name} success.")

        preds = model.predict(dataset)
        for messenger in messengers:
            messenger.send(preds)

        return preds

    def save_model(self):
        for tree in self.trees:
            self._removing_useless_message(tree.root)
        self.logger.log(f"Remove useless information done.")

        model_params = [(tree.record, tree.root) for tree in self.trees]
        saved_data = [
            model_params,
            self.task,
            self.n_labels,
            self.learning_rate,
            self.feature_importance_info,
        ]
        NumpyModelIO.save(saved_data, self.model_dir, self.model_name)
        self.logger.log(f"Save model {self.model_name} success.")

    def load_model(
        self,
        model_name: str,
        model_path: str = "./models",
    ) -> None:
        (
            model_params,
            task,
            n_labels,
            learning_rate,
            feature_importance_info,
        ) = NumpyModelIO.load(model_path, model_name)

        self.task = (task,)
        self.n_labels = (n_labels,)
        self.learning_rate = (learning_rate,)
        self.trees = [
            DecisionTree(
                task=task,
                n_labels=n_labels,
                crypto_type="",
                crypto_system=None,
                messengers=self.messengers,
                logger=self.logger,
            )
            for _ in range(len(model_params))
        ]

        for i, (record, root) in enumerate(model_params):
            self.trees[i].record = record
            self.trees[i].root = root

        self.feature_importance_info = feature_importance_info
        self.logger.log(f"Load model {model_name} success.")

    def print_model_structure(self, tree_structure="VERTICAL"):
        tree_strs = PrintTree.get_tree_strs(
            trees=self.trees, tree_structure=tree_structure
        )

        for tree_id, tree_str in tree_strs.items():
            print(tree_id)
            print(tree_str)

    def feature_importances_(
        self,
        importance_type: str = "split",
    ) -> Dict[str, list]:
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

        if importance_type != "split":
            # The "gain" and "cover" indicators are calculated as mean values
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

    def _validate_tree(self, tree, testset):
        for messenger_id, messenger in enumerate(self.messengers):
            if self.messengers_validTag[messenger_id]:
                messenger.send(wrap_message("validate", content=True))

        update_pred = tree.predict(testset.features)

        for i, messenger in enumerate(self.messengers):
            if self.messengers_validTag[i]:
                messenger.send(wrap_message("validate finished", content=True))

        self.logger.log("validate finished")

        return update_pred

    def _merge_tree_info(self, feature_importance_info_tree):
        if feature_importance_info_tree is not None:
            for eval_way in ["split", "gain", "cover"]:
                for key in feature_importance_info_tree[eval_way].keys():
                    self.feature_importance_info[eval_way][
                        key
                    ] += feature_importance_info_tree[eval_way][key]

        self.logger.log("merge tree information done")

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
