import pickle
import numpy as np
import datetime
from linkefl.base import BaseModelComponent
from linkefl.common.const import Const
from linkefl.feature.woe import PassiveWoe, TestWoe
from linkefl.vfl.linear import BaseLinearPassive
from linkefl.modelio import NumpyModelIO

class PassiveCreditCard(BaseLinearPassive, BaseModelComponent):
    def __init__(
        self,
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
        saving_model=True,
        model_path="./models",
        model_name=None,
    ):
        super(PassiveCreditCard, self).__init__(
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
            task="classification",
        )
        if model_name is None:
            self.model_name = (
                "vfl_{model_type}/{time}-{role}".format(
                    time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    role=Const.PASSIVE_NAME,
                    model_type='creditcard',
                )
                + ".model"
            )
        else:
            self.model_name = model_name

    def fit(self, trainset, validset, role=Const.PASSIVE_NAME):
        train_woe = PassiveWoe(trainset, [i for i in range(trainset.n_features)], _messenger)
        bin_bounds, bin_woe, bin_iv = train_woe.cal_woe()
        test_woe = TestWoe(
            validset, [i for i in range(trainset.n_features)], _messenger, bin_bounds, bin_woe
        )
        test_woe.cal_woe()
        with open(self.model_path+'/'+self.model_name[0:-12]+'-bounds.pkl', 'wb') as f1:
            pickle.dump(bin_bounds, f1)
        with open(self.model_path+'/'+self.model_name[0:-12]+'-woe.pkl', 'wb') as f2:
            pickle.dump(bin_woe, f2)
        self.train(trainset, validset)

    def score(self, testset, role=Const.PASSIVE_NAME):
        return self.predict(testset)
    
    @staticmethod
    def online_inference(
        dataset, model_name, messenger, model_path="./models"
    ):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        model_params = NumpyModelIO.load(model_path, model_name)
        with open(model_path+'/'+model_name[0:-12]+'-bounds.pkl', 'rb') as f1:
            bin_bounds = pickle.load(f1)
        with open(model_path+'/'+model_name[0:-12]+'-woe.pkl', 'rb') as f2:
            bin_woe = pickle.load(f2)
        test_woe = TestWoe(
            dataset, [i for i in range(dataset.n_features)], _messenger, bin_bounds, bin_woe
        )
        test_woe.cal_woe()
        # Scorecard

        p = 20 / np.log(2)  # 比例因子
        q = 600 - 20 * np.log(50) / np.log(2)  # 等于offset,偏移量
        test_score = np.zeros_like(dataset.features)
        test_score = np.around(model_params * dataset.features * p)
        print(test_score)
        print(np.amax(test_score, axis=0), np.amin(test_score, axis=0))
        pass_score = np.sum(test_score, axis=1)
        _messenger.send(pass_score)

        return None



if __name__ == "__main__":
    from linkefl.common.factory import logger_factory, messenger_factory
    from linkefl.dataio import NumpyDataset
    from linkefl.feature.transform import scale

    # 0. Set parameters
    _dataset_name = "credit"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = "localhost"
    active_port = 20002
    passive_ip = "localhost"
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
    _messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    _messenger.send("begin")
    # 1. Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print("Loading dataset...")
    passive_trainset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_testset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=False,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    # raise()
    passive_trainset = scale(passive_trainset)
    passive_testset = scale(passive_testset)
    # print(passive_trainset.features.shape, passive_testset.features.shape)

    # print(passive_trainset.features.shape, passive_testset.features.shape)
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    passive_party = PassiveCreditCard(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messenger=_messenger,
        crypto_type=_crypto_type,
        logger=_logger,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        using_pool=_using_pool,
        saving_model=True,
    )

    # passive_party.fit(passive_trainset, passive_testset)
    passive_party.online_inference(passive_testset, 'vfl_creditcard/20230314_170335-passive_party.model', _messenger)
    _messenger.close()
