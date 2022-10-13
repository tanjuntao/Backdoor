import datetime
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
        self.saving_model = saving_model
        self.model_path = model_path

        self.logger = logger_factory(role=Const.PASSIVE_NAME)

        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.PASSIVE_NAME,
            model_type=Const.VERTICAL_SBT,
        )

        self.feature_importance_info = {
            "split": defaultdict(int),  # Total number of splits
            "cover": defaultdict(int)  # Total sample covered
        }

        # given when training starts
        self.bin_index = None
        self.bin_split = None
        self.gh_recv = None
        self.compress = None
        self.capacity = None
        self.padding = None

        # filled as training goes on
        self.record = None

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
            if data["name"] == "train finished" and data["content"] is True:
                if self.saving_model:
                    model_name = (
                        f"{self.model_name}-{trainset.n_samples}_samples.model"
                    )
                    NumpyModelIO.save(self.record, self.model_path, model_name)
                self.logger.log("train finished")
                break
            elif data["name"] == "gh":
                self.gh_recv, self.compress, self.capacity, self.padding = data[
                    "content"
                ]
                self.logger.log("start a new tree")
            elif data["name"] == "record":
                feature_id, split_id, sample_tag = data["content"]
                record_id, sample_tag_left = self._save_record(
                    feature_id, split_id, sample_tag
                )
                self.logger.log(f"threshold saved in record_id: {record_id}")
                self.messenger.send(
                    wrap_message("record", content=(record_id, sample_tag_left))
                )
                # store feature split information
                self.feature_importance_info['split'][f'feature {feature_id}'] += 1
                self.feature_importance_info['cover'][f'feature {feature_id}'] += sum(sample_tag)
            elif data["name"] == "hist":
                sample_tag = data["content"]
                bin_gh_data = self._get_hist(sample_tag)
                # self.logger.log("bin_gh computed")
                self.messenger.send(wrap_message("hist", content=bin_gh_data))
            elif data["name"] == "validate" and data["content"] is True:
                self._validate(testset)
            else:
                raise KeyError

        self.logger.log(
            "Total training and validation time: {:.2f}".format(
                time.time() - start_time
            )
        )

    def _save_record(self, feature_id, split_id, sample_tag):
        record = np.array(
            [feature_id, self.bin_split[feature_id][split_id]]
        ).reshape(1, 2)

        if self.record is None:
            self.record = record
        else:
            self.record = np.concatenate((self.record, record), axis=0)

        record_id = len(self.record) - 1

        sample_tag_left = sample_tag
        sample_tag_left[self.bin_index[:, feature_id].flatten() > split_id] = 0

        return record_id, sample_tag_left

    def _get_hist(self, sample_tag):
        hist = PassiveHist(
            task=self.task,
            sample_tag=sample_tag,
            bin_index=self.bin_index,
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

    def predict(self, testset):
        self._validate(testset)

    def online_inference(self, dataset, model_name, model_path="./models"):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        self.record = NumpyModelIO.load(model_path, model_name)

        self._validate(dataset)

    def feature_importances_(self, evaluation_way="split"):
        assert evaluation_way in ("split", "cover"), "Not support evaluation way"

        keys = np.array(list(self.feature_importance_info[evaluation_way].keys()))
        values = np.array(list(self.feature_importance_info[evaluation_way].values()))

        ascend_index = values.argsort()
        keys, values = keys[ascend_index[::-1]], values[::-1]
        result = np.concatenate([keys.reshape((len(keys), -1)), values.reshape((len(values), -1))], axis=1)

        return result