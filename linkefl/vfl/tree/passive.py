import datetime
import time

import numpy as np
from termcolor import colored

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.messenger.base import Messenger
from linkefl.modelio import NumpyModelIO
from linkefl.vfl.tree.hist import PassiveHist
from linkefl.vfl.tree.data_functions import get_bin_info, wrap_message


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

        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.PASSIVE_NAME,
            model_type=Const.VERTICAL_SBT,
        )

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
        assert isinstance(trainset, NumpyDataset), "trainset should be an instance of NumpyDataset"
        assert isinstance(testset, NumpyDataset), "testset should be an instance of NumpyDataset"

        start_time = time.time()

        print("Building hist...")
        self.bin_index, self.bin_split = get_bin_info(trainset.features, self.max_bin)
        print("Done")

        print(colored("Waiting for active party...", "red"))

        while True:
            # ready to receive instructions from active party
            data = self.messenger.recv()
            if data["name"] == "train finished" and data["content"] is True:
                if self.saving_model:
                    model_name = self.model_name + "-" + str(trainset.n_samples) + "_samples" + ".model"
                    NumpyModelIO.save(self.record, self.model_path, model_name)
                print("train finished")
                break
            elif data["name"] == "gh":
                self.gh_recv, self.compress, self.capacity, self.padding = data["content"]
                print("\nstart a new tree")
            elif data["name"] == "record":
                feature_id, split_id, sample_tag = data["content"]
                record_id, sample_tag_left = self._save_record(feature_id, split_id, sample_tag)
                print(f"threshold saved in record_id: {record_id}")
                self.messenger.send(wrap_message("record", content=(record_id, sample_tag_left)))
            elif data["name"] == "hist":
                sample_tag = data["content"]
                bin_gh_data = self._get_hist(sample_tag)
                # print("bin_gh computed")
                self.messenger.send(wrap_message("hist", content=bin_gh_data))
            else:
                raise KeyError

        self._validate(testset)
        print(colored("Total training and validation time: {:.2f}".format(time.time() - start_time), "red"))

    def _save_record(self, feature_id, split_id, sample_tag):
        record = np.array([feature_id, self.bin_split[feature_id][split_id]]).reshape(1, 2)

        if self.record is None:
            self.record = record
        else:
            self.record = np.concatenate((self.record, record), axis=0)

        record_id = len(self.record) - 1

        sample_tag_left = sample_tag
        sample_tag_left[self.bin_index[:, feature_id].flatten() > split_id] = 0

        return record_id, sample_tag_left

    def _get_hist(self, sample_tag):
        hist = PassiveHist(task=self.task, sample_tag=sample_tag, bin_index=self.bin_index, gh_data=self.gh_recv)

        if self.task == "binary" and self.crypto_type == Const.PAILLIER and self.compress:
            return hist.compress(self.capacity, self.padding)
        else:
            return hist.bin_gh_data

    def _validate(self, testset):
        assert isinstance(testset, NumpyDataset), "testset should be an instance of NumpyDataset"

        features = testset.features

        while True:
            # ready to receive instructions from active party
            data = self.messenger.recv()
            if data["name"] == "validate finished" and data["content"] is True:
                print("validate finished")
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
        result = True if feature[int(feature_id)] > threshold else False  # avoid numpy bool

        return result

    def predict(self, testset):
        self._validate(testset)

    def online_inference(self, dataset, model_name, model_path="./models"):
        assert isinstance(dataset, NumpyDataset), "inference dataset should be an instance of NumpyDataset"
        self.record = NumpyModelIO.load(model_path, model_name)

        self._validate(dataset)
