import datetime
import os
import pathlib
import random
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, Optional

import numpy as np

from linkefl.base import BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO
from linkefl.vfl.tree.data_functions import (
    get_bin_info,
    get_latest_filename,
    wrap_message,
)
from linkefl.vfl.tree.hist import PassiveHist


class PassiveTreeParty(BaseModelComponent):
    def __init__(
        self,
        task: str,
        crypto_type: str,
        messenger: BaseMessenger,
        logger: GlobalLogger,
        *,
        max_bin: int = 16,
        colsample_bytree: int = 1,
        n_processes: int = 1,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Passive Tree Party class to train and validate dataset

        Args:
            task: binary or multi
            max_bin: max bin number for a feature point
        """
        self.task = task
        self.crypto_type = crypto_type
        self.messenger = messenger
        self.logger = logger

        self.max_bin = max_bin
        self.colsample_bytree = colsample_bytree
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
                        role=Const.PASSIVE_NAME,
                        algo_name=Const.AlgoNames.VFL_SBT,
                    )
                    + ".model"
                )
            self.model_dir = model_dir
            self.model_name = model_name
            if not os.path.exists(self.model_dir):
                pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.feature_importance_info = {
            "split": defaultdict(int),  # Total number of splits
            "cover": defaultdict(float),  # Total sample covered
        }

        self.pool = Pool(n_processes)

        # given when training starts
        self.bin_index = None
        self.bin_split = None
        self.gh_recv = None
        self.compress = None
        self.capacity = None
        self.padding = None
        self.feature_index_selected = None
        self.bin_index_selected = None

        # filled as training goes on
        self.record = None

    def fit(
        self,
        trainset: NumpyDataset,
        validset: NumpyDataset,
        role: str = Const.PASSIVE_NAME,
    ) -> None:
        self.train(trainset, validset)

    def train(self, trainset: NumpyDataset, testset: NumpyDataset) -> None:
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        start_time = time.time()

        self.logger.log("Building hist...")
        self.bin_index, self.bin_split = get_bin_info(
            trainset.features, self.max_bin, self.pool
        )
        self.logger.log("Done")

        self.logger.log("Waiting for active party...")

        while True:
            # ready to receive instructions from active party
            data = self.messenger.recv()

            if data["name"] == "gh":
                self.gh_recv, self.compress, self.capacity, self.padding = data[
                    "content"
                ]
                self._init_tree_info()  # information for building a new tree
                self.logger.log("start a new tree")

                self.logger.log_component(
                    name=Const.VERTICAL_SBT,
                    status=Const.RUNNING,
                    begin=start_time,
                    end=time.time(),
                    duration=time.time() - start_time,
                    progress=0.5,  # passive party does not know the total tree number
                )

            elif data["name"] == "hist":
                sample_tag = data["content"]
                bin_gh_data = self._get_hist(sample_tag)
                self.logger.log("bin_gh computed")
                self.messenger.send(wrap_message("hist", content=bin_gh_data))

            elif data["name"] == "record":
                feature_id, split_id, sample_tag_selected, sample_tag_unselected = data[
                    "content"
                ]

                (
                    feature_id_origin,
                    record_id,
                    sample_tag_selected_left,
                    sample_tag_unselected_left,
                ) = self._save_record(
                    feature_id, split_id, sample_tag_selected, sample_tag_unselected
                )
                self.logger.log(f"threshold saved in record_id: {record_id}")
                self.messenger.send(
                    wrap_message(
                        "record",
                        content=(
                            feature_id_origin,
                            record_id,
                            sample_tag_selected_left,
                            sample_tag_unselected_left,
                        ),
                    )
                )

            elif data["name"] == "validate" and data["content"] is True:
                # after training a tree, enter the evaluation stage
                self._merge_tree_info()  # merge information from current tree

                # store temp file
                if self.saving_model:
                    # model_name = (
                    #     f"{self.model_name}-{trainset.n_samples}_samples.model"
                    # )
                    model_name = self.model_name
                    NumpyModelIO.save(
                        [self.record, self.feature_importance_info],
                        self.model_dir,
                        model_name,
                    )

                self.logger.log("temp model saved")
                self._validate(testset)

            elif data["name"] == "train finished" and data["content"] is True:
                if self.saving_model:
                    # model_name = (
                    #     f"{self.model_name}-{trainset.n_samples}_samples.model"
                    # )
                    model_structure = self.messenger.recv()
                    model_name = self.model_name
                    NumpyModelIO.save(
                        [self.record, self.feature_importance_info, model_structure],
                        self.model_dir,
                        model_name,
                    )
                self.logger.log_component(
                    name=Const.VERTICAL_SBT,
                    status=Const.SUCCESS,
                    begin=start_time,
                    end=time.time(),
                    duration=time.time() - start_time,
                    progress=1.0,
                )
                self.logger.log("train finished")
                break

            else:
                raise KeyError

        if self.pool is not None:
            self.pool.close()

        self.logger.log(
            "Total training and validation time: {:.2f}".format(
                time.time() - start_time
            )
        )

    def load_retrain(
        self,
        load_model_path: str,
        trainset: NumpyDataset,
        testset: NumpyDataset,
    ) -> None:
        """breakpoint retraining function."""
        model_name = get_latest_filename(load_model_path)
        self.record, self.feature_importance_info, model_structure = NumpyModelIO.load(
            load_model_path, model_name
        )
        self.train(trainset, testset)

    def _save_record(
        self, feature_id, split_id, sample_tag_selected, sample_tag_unselected
    ):
        feature_id_origin = self.feature_index_selected[feature_id]

        # store feature split information
        self.feature_importance_info_tree["split"][f"feature{feature_id_origin}"] += 1
        self.feature_importance_info_tree["cover"][
            f"feature{feature_id_origin}"
        ] += sum(sample_tag_selected)
        self.logger.log("store feature split information")

        # update record
        record = np.array(
            [feature_id_origin, self.bin_split[feature_id_origin][split_id]]
        ).reshape(1, 2)

        if self.record_tree is None:
            self.record_tree = record
        else:
            self.record_tree = np.concatenate((self.record_tree, record), axis=0)

        if self.record is None:
            record_id = len(self.record_tree) - 1
        else:
            record_id = len(self.record) + len(self.record_tree) - 1

        # update sample_tag
        sample_tag_selected_left = sample_tag_selected
        sample_tag_selected_left[
            self.bin_index[:, feature_id_origin].flatten() > split_id
        ] = 0
        sample_tag_unselected_left = sample_tag_unselected
        sample_tag_unselected_left[
            self.bin_index[:, feature_id_origin].flatten() > split_id
        ] = 0

        return (
            feature_id_origin,
            record_id,
            sample_tag_selected_left,
            sample_tag_unselected_left,
        )

    def _get_hist(self, sample_tag):
        hist = PassiveHist(
            task=self.task,
            sample_tag=sample_tag,
            bin_index=self.bin_index_selected,
            gh_data=self.gh_recv,
        )

        if (
            self.task == "binary"
            and self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER)
            and self.compress
        ):
            return hist.compress(self.capacity, self.padding)
        else:
            return hist.bin_gh_data

    def _validate(self, testset):
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        features = testset.features

        while True:
            # ready to receive instructions from active party
            data = self.messenger.recv()
            if data["name"] == "validate finished" and data["content"] is True:
                self.logger.log("validate finished")
                break
            elif data["name"] == "judge":
                sample_id, record_id = data["content"]
                result = self._judge(features[sample_id], record_id)
                self.messenger.send(wrap_message("judge", content=result))
            else:
                raise KeyError

    def _judge(self, feature, record_id):
        feature_id, threshold = self.record[record_id]
        # feature_id is float thanks to threshold...
        result = (
            True if feature[int(feature_id)] > threshold else False
        )  # avoid numpy bool

        return result

    def score(
        self,
        testset: NumpyDataset,
        role: str = Const.PASSIVE_NAME,
    ) -> None:
        self.predict(testset)

    def predict(self, testset: NumpyDataset) -> None:
        self._validate(testset)

    @staticmethod
    def online_inference(
            dataset: NumpyDataset,
            messenger: BaseMessenger,
            logger: GlobalLogger,
            model_dir: str,
            model_name: str,
            role: str = Const.PASSIVE_NAME,
    ):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        record, feature_importance_info, model_structure = NumpyModelIO.load(
            model_dir, model_name
        )

        features = dataset.features

        while True:
            # ready to receive instructions from active party
            data = messenger.recv()
            if data["name"] == "validate finished" and data["content"] is True:
                logger.log("validate finished")
                break
            elif data["name"] == "judge":
                sample_id, record_id = data["content"]

                feature_id, threshold = record[record_id]
                # feature_id is float thanks to threshold...
                result = (
                    True if features[sample_id][int(feature_id)] > threshold else False
                )  # avoid numpy bool

                messenger.send(wrap_message("judge", content=result))
            else:
                raise KeyError

        preds = messenger.recv()
        return preds

    def _init_tree_info(self):
        """Initialize the tree-level information store"""
        self.record_tree = None
        self.feature_importance_info_tree = {
            "split": defaultdict(int),  # Total number of splits
            "cover": defaultdict(float),  # Total sample covered
        }

        # perform feature selection
        feature_num = self.bin_index.shape[1]
        self.feature_index_selected = random.sample(
            list(range(feature_num)), int(feature_num * self.colsample_bytree)
        )
        self.bin_index_selected = np.array(self.bin_index.copy())
        self.bin_index_selected = self.bin_index_selected[
            :, self.feature_index_selected
        ]

        self.logger.log("init tree information and feature selection done")

    def _merge_tree_info(self):
        """Merge information from a single tree"""
        # merge record message
        if self.record is None:
            self.record = self.record_tree
        elif self.record_tree is None:
            self.record = self.record
        else:
            self.record = np.concatenate((self.record, self.record_tree), axis=0)

        # merge feature importance info
        if self.feature_importance_info_tree is not None:
            for key in self.feature_importance_info_tree["split"].keys():
                self.feature_importance_info["split"][
                    key
                ] += self.feature_importance_info_tree["split"][key]
            for key in self.feature_importance_info_tree["cover"].keys():
                self.feature_importance_info["cover"][
                    key
                ] += self.feature_importance_info_tree["cover"][key]

        # clear temporary information
        self.record_tree, self.feature_importance_info_tree = None, None
        self.logger.log("merge tree information done")

    def feature_importances_(self, importance_type: str = "split") -> Dict[str, list]:
        assert importance_type in ("split", "cover"), "Not support importance type"

        keys = np.array(list(self.feature_importance_info[importance_type].keys()))
        values = np.array(list(self.feature_importance_info[importance_type].values()))

        if importance_type == "cover":
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
