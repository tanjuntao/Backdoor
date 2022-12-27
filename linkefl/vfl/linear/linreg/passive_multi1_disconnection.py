from linkefl.common.const import Const
from linkefl.common.factory import logger_factory, messenger_factory,messenger_factory_multi_disconnection
from linkefl.dataio import NumpyDataset
# from linkefl.vfl.linear import BaseLinearPassive
from  linkefl.vfl.linear.base_multi import BaseLinearPassive
from  linkefl.vfl.linear.linreg.multi_disconnection import PassiveLinReg_disconnection




if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'diabetes'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 20001
    _epochs = 200000
    _batch_size = -1
    _learning_rate = 1.0
    _penalty = Const.NONE
    _reg_lambda = 0.01
    _random_state = None
    _crypto_type = Const.PLAIN
    _val_freq = 5000
    saving_model = True

    # 1. Load datasets
    print('Loading dataset...')
    passive_trainset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
                                                    dataset_name=dataset_name,
                                                    root='../../data',
                                                    train=True,
                                                    download=True,
                                                    passive_feat_frac=passive_feat_frac,
                                                    feat_perm_option=feat_perm_option)
    passive_testset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
                                                   dataset_name=dataset_name,
                                                   root='../../data',
                                                   train=False,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    passive_trainset = NumpyDataset.feature_split(passive_trainset, n_splits=2)[0]
    passive_testset = NumpyDataset.feature_split(passive_testset, n_splits=2)[0]
    print('Done.')

    # 2. Dataset preprocessing
    # passive_trainset = scale(passive_trainset)
    # passive_testset = scale(passive_testset)

    # 3. Initialize messenger
    _messenger = messenger_factory_multi_disconnection(messenger_type=Const.FAST_SOCKET,
                                   role=Const.PASSIVE_NAME,
                                    model_type="NN",
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)

    # 4. Initialize model and start training
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    passive_party = PassiveLinReg_disconnection(epochs=_epochs,
                                  batch_size=_batch_size,
                                  learning_rate=_learning_rate,
                                  messenger=_messenger,
                                  crypto_type=_crypto_type,
                                  logger=_logger,
                                  penalty=_penalty,
                                  reg_lambda=_reg_lambda,
                                  random_state=_random_state,
                                  val_freq=_val_freq,
                                  saving_model=saving_model)

    passive_party.train(passive_trainset, passive_testset)

    # 5. Close messenger, finish training
    _messenger.close()

    # # For online inference, you only need to substitue the model name
    # scores = PassiveLinReg.online_inference(
    #     passive_testset,
    #     model_name='20220831_190255-passive_party-vertical_linreg-402_samples.model',
    #     messenger=_messenger
    # )
    # print(scores)
