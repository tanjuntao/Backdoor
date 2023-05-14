import math
from typing import Dict, List, Optional

import numpy as np

from linkefl.base import BaseCryptoSystem, BaseMessenger
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.feature.woe import ActiveWoe, TestWoe
from linkefl.modelio import NumpyModelIO
from linkefl.vfl.linear.logreg import ActiveLogReg


class ActiveCreditCard(ActiveLogReg):
    def __init__(
        self,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        messengers: List[BaseMessenger],
        cryptosystem: BaseCryptoSystem,
        logger: GlobalLogger,
        rank: int = 0,
        p: float = 20 / math.log(2),
        q: float = 600 - 20 * math.log(50) / math.log(2),
        penalty: str = "l2",
        reg_lambda: float = 0.01,
        num_workers: int = 1,
        val_freq: int = 1,
        random_state: Optional[int] = None,
        residue_precision: float = 0.0001,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        positive_thresh: float = 0.5,
        ks_cut_points: int = 50,
    ):
        super(ActiveCreditCard, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messengers=messengers,
            cryptosystem=cryptosystem,
            logger=logger,
            rank=rank,
            penalty=penalty,
            reg_lambda=reg_lambda,
            num_workers=num_workers,
            val_freq=val_freq,
            random_state=random_state,
            residue_precision=residue_precision,
            saving_model=saving_model,
            model_dir=model_dir,
            model_name=model_name,
            positive_thresh=positive_thresh,
            ks_cut_points=ks_cut_points,
        )
        self.p: float = p
        self.q: float = q

        if self.saving_model:
            if model_name is None:
                algo_name = Const.AlgoNames.VFL_CREDITCARD
                self.model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.ACTIVE_NAME,
                        algo_name=algo_name,
                    )
                    + ".model"
                )

    def validate(self, validset: NumpyDataset) -> Dict[str, float]:
        scores = super(ActiveCreditCard, self).validate(validset)
        params = getattr(self, "params")
        base_credits = round(self.q + self.p * params[-1], 0)
        valid_credits = np.around(params * validset.features * self.p)
        active_credits = np.sum(valid_credits[:, 0:-1], axis=1)
        for msger in self.messengers:
            passive_credits = msger.recv()
            active_credits += passive_credits
        final_credits = active_credits + base_credits
        scores.update({"credits": final_credits})
        return scores

    @staticmethod
    def online_inference(
        dataset: NumpyDataset,
        messengers: List[BaseMessenger],
        logger: GlobalLogger,
        model_dir: str,
        model_name: str,
        positive_thresh: float = 0.5,
        role: str = Const.ACTIVE_NAME,
        p: float = 20 / math.log(2),
        q: float = 600 - 20 * math.log(50) / math.log(2),
    ):
        params = NumpyModelIO.load(model_dir, model_name)
        base_credits = round(q + p * params[-1], 0)
        valid_credits = np.around(params * dataset.features * p)
        active_credits = np.sum(valid_credits[:, 0:-1], axis=1)
        for msger in messengers:
            passive_credits = msger.recv()
            active_credits += passive_credits
        final_credits = active_credits + base_credits

        return final_credits


if __name__ == "__main__":
    from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
    from linkefl.feature.transform import add_intercept, parse_label, scale

    # Set parameters
    _dataset_name = "credit"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _active_ips = [
        "localhost",
    ]
    _active_ports = [
        20000,
    ]
    _passive_ips = [
        "localhost",
    ]
    _passive_ports = [
        30000,
    ]
    _epochs = 100
    _batch_size = -1
    _learning_rate = 0.1
    _penalty = Const.L2
    _reg_lambda = 0.01
    _crypto_type = Const.PLAIN
    _random_state = 3347
    _key_size = 1024
    _num_workers = 1
    _saving_model = True
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=10,
        gen_from_set=False,
    )
    _messengers = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=ac_ip,
            active_port=ac_port,
            passive_ip=pass_ip,
            passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in zip(
            _active_ips, _active_ports, _passive_ips, _passive_ports
        )
    ]

    # Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=True,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
    )
    active_testset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=False,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
    )
    # print(active_trainset.features.shape, active_testset.features.shape)
    active_trainset = scale(parse_label(active_trainset))
    active_testset = scale(parse_label(active_testset))

    active_woe = ActiveWoe(active_trainset, [0, 1, 2, 3, 4], _messengers)
    bin_bounds, bin_woe, bin_iv = active_woe.cal_woe()
    active_trainset = add_intercept(active_trainset)
    test_woe = TestWoe(
        active_testset, [0, 1, 2, 3, 4], _messengers, bin_bounds, bin_woe
    )
    test_woe.cal_woe()
    active_testset = add_intercept(active_testset)
    print(active_trainset.features.shape, active_testset.features.shape)

    # Initialize model and start training
    print("ACTIVE PARTY started, listening...")
    active_party = ActiveCreditCard(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messengers=_messengers,
        cryptosystem=_crypto,
        logger=_logger,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        saving_model=_saving_model,
    )
    active_party.train(active_trainset, active_testset)

    # Close messeger
    for msger_ in _messengers:
        msger_.close()
