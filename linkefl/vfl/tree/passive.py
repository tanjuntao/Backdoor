import time

import numpy as np
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import messenger_factory
from linkefl.dataio import NumpyDataset
from linkefl.messenger.base import Messenger
from linkefl.vfl.tree.hist import PassiveHist
from linkefl.vfl.tree.data_functions import get_bin_info, wrap_message


class PassiveTreeParty:
    def __init__(self, task: str, crypto_type: str, messenger: Messenger, *, max_bin: int = 16):
        """Passive Tree Party class to train and validate dataset

        Args:
            task: binary or multi
            max_bin: max bin number for a feature point
        """

        self.task = task
        self.crypto_type = crypto_type
        self.messenger = messenger

        self.max_bin = max_bin

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
                print("train finished")
                break
            elif data["name"] == "gh":
                self.gh_recv, self.compress, self.capacity, self.padding = data["content"]
                print("\nstart a new tree")
            elif data["name"] == "record":
                _, feature_id, split_id, sample_tag = data["content"]
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
                _, sample_id, record_id = data["content"]
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


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "wine"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    task = "multi"
    _crypto_type = Const.PAILLIER

    active_ip = "localhost"
    active_port = 20000
    passive_ip = "localhost"
    passive_port = 30000

    # 1. Load datasets
    print("Loading dataset...")
    passive_trainset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=dataset_name,
        train=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_testset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=dataset_name,
        train=False,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    print("Done")

    # 2. Initialize messenger
    messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    # 3. Initialize passive tree party and start training
    passive_party = PassiveTreeParty(task=task, crypto_type=_crypto_type, messenger=messenger)
    passive_party.train(passive_trainset, passive_testset)

    # 4. Close messenger, finish training
    messenger.close()
