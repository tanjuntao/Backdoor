import datetime
import random
import time

import numpy as np
from collections import defaultdict

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import NumpyDataset
from linkefl.messenger.base import Messenger
from linkefl.modelio import NumpyModelIO
from linkefl.vfl.tree.data_functions import get_bin_info, wrap_message
from linkefl.vfl.tree.hist import PassiveHist


class PassiveTreeParty:
    def __init__(
        self,
        task: str,
        crypto_type: str,
        messenger: Messenger,
        *,
        max_bin: int = 16,
        colsample_bytree = 1,
        saving_model: bool = False,
        model_path: str = "./models",
    ):
        """Passive Tree Party class to train and validate dataset

        Args:
            task: binary or multi
            max_bin: max bin number for a feature point
        """

        self.task = task
        self.crypto_type = crypto_type
        self.messenger = messenger

        self.max_bin = max_bin
        self.colsample_bytree = colsample_bytree
        self.saving_model = saving_model
        self.model_path = model_path

        self.logger = logger_factory(role=Const.PASSIVE_NAME)

        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.PASSIVE_NAME,
            model_type=Const.VERTICAL_SBT,
        )

        self.feature_importance_info = {
            "split": defaultdict(int),      # Total number of splits
            "cover": defaultdict(float)       # Total sample covered
        }

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

    def fit(self, trainset, testset, role=Const.PASSIVE_NAME):
        self.train(trainset, testset)

    def train(self, trainset, testset):
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"

        start_time = time.time()

        self.logger.log("Building hist...")
        self.bin_index, self.bin_split = get_bin_info(
            trainset.features, self.max_bin
        )
        self.logger.log("Done")

        self.logger.log("Waiting for active party...")

        while True:
            # ready to receive instructions from active party
            data = self.messenger.recv()

            if data["name"] == "gh":
                self.gh_recv, self.compress, self.capacity, self.padding = data["content"]
                self._init_tree_info()      # information for building a new tree
                self.logger.log("start a new tree")

            elif data["name"] == "hist":
                sample_tag = data["content"]
                bin_gh_data = self._get_hist(sample_tag)
                self.logger.log("bin_gh computed")
                self.messenger.send(wrap_message("hist", content=bin_gh_data))

            elif data["name"] == "record":
                feature_id, split_id, sample_tag_selected, sample_tag_unselected = data["content"]

                feature_id_origin, record_id, sample_tag_selected_left, sample_tag_unselected_left = self._save_record(
                    feature_id, split_id, sample_tag_selected, sample_tag_unselected
                )
                self.logger.log(f"threshold saved in record_id: {record_id}")
                self.messenger.send(
                    wrap_message("record", content=(feature_id_origin, record_id, sample_tag_selected_left, sample_tag_unselected_left))
                )

            elif data["name"] == "validate" and data["content"] is True:
                # after training a tree, enter the evaluation stage
                self._merge_tree_info()     # merge information from current tree

                # store temp file
                if self.saving_model:
                    model_name = (
                        f"{self.model_name}-{trainset.n_samples}_samples.model"
                    )
                NumpyModelIO.save([self.record, self.feature_importance_info], self.model_path, model_name)

                self.logger.log("temp model saved")
                self._validate(testset)

            elif data["name"] == "train finished" and data["content"] is True:
                if self.saving_model:
                    model_name = (
                        f"{self.model_name}-{trainset.n_samples}_samples.model"
                    )
                    NumpyModelIO.save([self.record, self.feature_importance_info], self.model_path, model_name)
                self.logger.log("train finished")
                break

            else:
                raise KeyError

        self.logger.log(
            "Total training and validation time: {:.2f}".format(
                time.time() - start_time
            )
        )

    def load_retrain(self, load_model_path, load_model_name, trainset, testset):
        """breakpoint retraining function.
        """
        self.record, self.feature_importance_info = NumpyModelIO.load(load_model_path, load_model_name)
        self.train(trainset, testset)

    def _save_record(self, feature_id, split_id, sample_tag_selected, sample_tag_unselected):
        feature_id_origin = self.feature_index_selected[feature_id]

        # store feature split information
        self.feature_importance_info_tree['split'][f'feature{feature_id_origin}'] += 1
        self.feature_importance_info_tree['cover'][f'feature{feature_id_origin}'] += sum(sample_tag_selected)
        self.logger.log(f"store feature split information")

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
        sample_tag_selected_left[self.bin_index[:, feature_id_origin].flatten() > split_id] = 0
        sample_tag_unselected_left = sample_tag_unselected
        sample_tag_unselected_left[self.bin_index[:, feature_id_origin].flatten() > split_id] = 0

        return feature_id_origin, record_id, sample_tag_selected_left , sample_tag_unselected_left

    def _get_hist(self, sample_tag):
        hist = PassiveHist(
            task=self.task,
            sample_tag=sample_tag,
            bin_index=self.bin_index_selected,
            gh_data=self.gh_recv,
        )

        if (
            self.task == "binary"
            and self.crypto_type == Const.PAILLIER
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

    def score(self, testset, role=Const.PASSIVE_NAME):
        self.predict(testset)

    def predict(self, testset):
        self._validate(testset)

    def online_inference(self, dataset, model_name, model_path="./models"):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        self.record, self.feature_importance_info = NumpyModelIO.load(model_path, model_name)

        self._validate(dataset)

    def _init_tree_info(self):
        """Initialize the tree-level information store
        """
        self.record_tree = None
        self.feature_importance_info_tree = {
            "split": defaultdict(int),        # Total number of splits
            "cover": defaultdict(float)       # Total sample covered
        }

        # perform feature selection
        feature_num = self.bin_index.shape[1]
        self.feature_index_selected = random.sample(list(range(feature_num)), int(feature_num * self.colsample_bytree))
        self.bin_index_selected = np.array(self.bin_index.copy())
        self.bin_index_selected = self.bin_index_selected[:, self.feature_index_selected]

        self.logger.log("init tree information and feature selection done")

    def _merge_tree_info(self):
        """Merge information from a single tree
        """
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
                self.feature_importance_info["split"][key] += self.feature_importance_info_tree["split"][key]
            for key in self.feature_importance_info_tree["cover"].keys():
                self.feature_importance_info["cover"][key] += self.feature_importance_info_tree["cover"][key]

        # clear temporary information
        self.record_tree, self.feature_importance_info_tree = None, None
        self.logger.log("merge tree information done")

    def feature_importances_(self, importance_type="split"):
        assert importance_type in ("split", "cover"), "Not support importance type"

        keys = np.array(list(self.feature_importance_info[importance_type].keys()))
        values = np.array(list(self.feature_importance_info[importance_type].values()))

        if importance_type == 'cover':
            split_nums = np.array(list(self.feature_importance_info['split'].values()))
            split_nums[split_nums == 0] = 1  # Avoid division by zero
            values = values / split_nums

        ascend_index = values.argsort()
        features, values = keys[ascend_index[::-1]], values[ascend_index[::-1]]
        result = {
            'features': list(features),
            f'importance_{importance_type}': list(values)
        }

        return result