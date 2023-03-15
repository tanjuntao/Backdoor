from typing import List, Optional

from linkefl.base import BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.vfl.linear.base import BaseLinearPassive


class PassiveLinReg(BaseLinearPassive, BaseModelComponent):
    def __init__(
        self,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        messenger: BaseMessenger,
        logger: GlobalLogger,
        rank: int = 1,
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
        super(PassiveLinReg, self).__init__(
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
            task="regression",
        )

    def fit(
        self,
        trainset: NumpyDataset,
        validset: NumpyDataset,
        role: str = Const.PASSIVE_NAME,
    ) -> None:
        self.train(trainset, validset)

    def score(
        self,
        testset: NumpyDataset,
        role: str = Const.PASSIVE_NAME,
    ) -> None:
        return self.predict(testset)


if __name__ == "__main__":
    from linkefl.common.factory import logger_factory, messenger_factory
    from linkefl.dataio import NumpyDataset

    # Set parameters
    _dataset_name = "diabetes"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _active_ip = "localhost"
    _active_port = 20000
    _passive_ip = "localhost"
    _passive_port = 30000
    _epochs = 200000
    _batch_size = -1
    _learning_rate = 1.0
    _penalty = Const.NONE
    _reg_lambda = 0.01
    _random_state = None
    _val_freq = 5000
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

    # Load datasets
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
    # passive_trainset = NumpyDataset.feature_split(passive_trainset, n_splits=2)[0]
    # passive_testset = NumpyDataset.feature_split(passive_testset, n_splits=2)[0]
    # passive_trainset = scale(passive_trainset)
    # passive_testset = scale(passive_testset)

    # Initialize model and start training
    passive_party = PassiveLinReg(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messenger=_messenger,
        logger=_logger,
        rank=1,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        num_workers=_num_workers,
        random_state=_random_state,
        val_freq=_val_freq,
        saving_model=_saving_model,
    )
    passive_party.train(passive_trainset, passive_testset)

    # Close messengers, finish training
    _messenger.close()
