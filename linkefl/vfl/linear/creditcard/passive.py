import numpy as np

from linkefl.base import BaseModelComponent
from linkefl.common.const import Const
from linkefl.vfl.linear import BaseLinearPassive
from linkefl.feature.woe import PassiveWoe, TestWoe


class PassiveLogReg(BaseLinearPassive, BaseModelComponent):
    def __init__(self,
                 epochs,
                 batch_size,
                 learning_rate,
                 messenger,
                 crypto_type,
                 logger,
                 *,
                 rank=1,
                 penalty=Const.L2,
                 reg_lambda=0.01,
                 precision=0.001,
                 random_state=None,
                 using_pool=False,
                 num_workers=-1,
                 val_freq=1,
                 saving_model=False,
                 model_path='./models',
                 model_name=None,
                 ):
        super(PassiveLogReg, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messenger=messenger,
            crypto_type=crypto_type,
            logger=logger,
            rank=rank,
            penalty=penalty,
            reg_lambda=reg_lambda,
            precision=precision,
            random_state=random_state,
            using_pool=using_pool,
            num_workers=num_workers,
            val_freq=val_freq,
            saving_model=saving_model,
            model_path=model_path,
            model_name=model_name,
            task='classification'
        )

    def fit(self, trainset, validset, role=Const.PASSIVE_NAME):
        self.train(trainset, validset)

    def score(self, testset, role=Const.PASSIVE_NAME):
        return self.predict(testset)


if __name__ == '__main__':
    from linkefl.common.factory import logger_factory, messenger_factory
    from linkefl.dataio import NumpyDataset
    from linkefl.feature.transform import scale

    # 0. Set parameters
    _dataset_name = 'credit'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 20003
    passive_ip = 'localhost'
    passive_port = 20001
    _epochs = 100
    _batch_size = -1
    _learning_rate = 0.1
    _penalty = Const.L2
    _reg_lambda = 0.01
    _random_state = 3347
    _crypto_type = Const.PLAIN
    _using_pool = False

    # 3. Initialize messenger
    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.PASSIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)

    # 1. Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print('Loading dataset...')
    passive_trainset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
                                                    dataset_name=_dataset_name,
                                                    root='../../data',
                                                    train=True,
                                                    download=True,
                                                    passive_feat_frac=passive_feat_frac,
                                                    feat_perm_option=feat_perm_option)
    passive_testset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
                                                   dataset_name=_dataset_name,
                                                   root='../../data',
                                                   train=False,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    # raise()
    passive_woe = PassiveWoe(passive_trainset, [0, 1, 2, 3, 4], _messenger)
    bin_bounds, bin_woe, bin_iv = passive_woe.cal_woe()

    test_woe = TestWoe(passive_testset, [0, 1, 2, 3, 4], _messenger, bin_bounds, bin_woe)
    test_woe.cal_woe()

    _logger = logger_factory(role=Const.PASSIVE_NAME)
    passive_party = PassiveLogReg(epochs=_epochs,
                                  batch_size=_batch_size,
                                  learning_rate=_learning_rate,
                                  messenger=_messenger,
                                  crypto_type=_crypto_type,
                                  logger=_logger,
                                  penalty=_penalty,
                                  reg_lambda=_reg_lambda,
                                  random_state=_random_state,
                                  using_pool=_using_pool,
                                  saving_model=False)

    passive_party.train(passive_trainset, passive_testset)
    w_p = passive_party.params
    print(w_p)

    p = 20 / np.log(2)  # 比例因子
    q = 600 - 20 * np.log(50) / np.log(2)  # 等于offset,偏移量
    test_score = np.zeros_like(passive_testset.features)
    test_score = np.around(w_p * passive_testset.features * p)
    print(test_score)
    print(np.amax(test_score, axis=0), np.amin(test_score, axis=0))
    pass_score = np.sum(test_score, axis=1)
    _messenger.send(pass_score)

    _messenger.close()