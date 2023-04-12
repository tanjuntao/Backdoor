import math
from typing import Optional

import numpy as np

from linkefl.base import BaseMessenger
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.feature.woe import PassiveWoe, TestWoe
from linkefl.modelio import NumpyModelIO
from linkefl.vfl.linear.logreg import PassiveLogReg


class PassiveCreditCard(PassiveLogReg):
    def __init__(
        self,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        messenger: BaseMessenger,
        logger: GlobalLogger,
        rank: int = 1,
        p: float = 20 / math.log(2),
        penalty: str = "l2",
        reg_lambda: float = 0.01,
        num_workers: int = 1,
        val_freq: int = 1,
        random_state: Optional[int] = None,
        encode_precision: float = 0.001,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        super(PassiveCreditCard, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messenger=messenger,
            logger=logger,
            rank=rank,
            penalty=penalty,
            reg_lambda=reg_lambda,
            num_workers=num_workers,
            val_freq=val_freq,
            random_state=random_state,
            encode_precision=encode_precision,
            saving_model=saving_model,
            model_dir=model_dir,
            model_name=model_name,
        )
        self.p: float = p

        if self.saving_model:
            if model_name is None:
                algo_name = Const.AlgoNames.VFL_CREDITCARD
                self.model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.PASSIVE_NAME,
                        algo_name=algo_name,
                    )
                    + ".model"
                )

    def validate(self, validset: NumpyDataset) -> None:
        super(PassiveCreditCard, self).validate(validset)
        valid_credits = np.around(getattr(self, "params") * validset.features * self.p)
        passive_credits = np.sum(valid_credits, axis=1)
        self.messenger.send(passive_credits)

    @staticmethod
    def online_inference(
        dataset: NumpyDataset,
        messenger: BaseMessenger,
        logger: GlobalLogger,
        model_dir: str,
        model_name: str,
        role: str = Const.PASSIVE_NAME,
        p: float = 20 / math.log(2),
    ):
        scores, _ = super(PassiveCreditCard, PassiveCreditCard).online_inference(
            dataset=dataset,
            messenger=messenger,
            logger=logger,
            model_dir=model_dir,
            model_name=model_name,
            role=role,
        )
        params = NumpyModelIO.load(model_dir, model_name)
        valid_credits = np.around(params * dataset.features * p)
        passive_credits = np.sum(valid_credits, axis=1)
        messenger.send(passive_credits)
        final_credits = messenger.recv()

        return scores, final_credits


if __name__ == "__main__":
    from linkefl.common.factory import logger_factory, messenger_factory
    from linkefl.dataio import NumpyDataset
    from linkefl.feature.transform import scale

    # 0. Set parameters
    _dataset_name = "credit"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _active_ip = "localhost"
    _active_port = 20000
    _passive_ip = "localhost"
    _passive_port = 30000
    _epochs = 100
    _batch_size = -1
    _learning_rate = 0.1
    _penalty = Const.L2
    _reg_lambda = 0.01
    _random_state = 3347
    _num_workers = 1
    _saving_model = True
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    _messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=_active_ip,
        active_port=_active_port,
        passive_ip=_passive_ip,
        passive_port=_passive_port,
    )

    # 1. Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print("Loading dataset...")
    passive_trainset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=True,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
    )
    passive_testset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=False,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
    )
    passive_trainset = scale(passive_trainset)
    passive_testset = scale(passive_testset)

    passive_woe = PassiveWoe(passive_trainset, [0, 1, 2, 3, 4], _messenger)
    bin_bounds, bin_woe, bin_iv = passive_woe.cal_woe()
    test_woe = TestWoe(
        passive_testset, [0, 1, 2, 3, 4], _messenger, bin_bounds, bin_woe
    )
    test_woe.cal_woe()
    print(passive_trainset.features.shape, passive_testset.features.shape)

    # Initialize model and start training
    passive_party = PassiveCreditCard(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messenger=_messenger,
        logger=_logger,
        rank=1,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        num_workers=_num_workers,
        saving_model=_saving_model,
    )
    passive_party.train(passive_trainset, passive_testset)

    # Close messenger
    _messenger.close()
