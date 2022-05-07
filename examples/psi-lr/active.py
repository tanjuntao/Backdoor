from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.dataio import NDArrayDataset
from linkefl.feature import scale, add_intercept



# 0. Set parameters
trainset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_train.csv'
testset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_test.csv'
passive_feat_frac = 0.5
feat_perm_option = Const.SEQUENCE
active_ip = 'localhost'
active_port = 20000
passive_ip = 'localhost'
passive_port = 30000
_epochs = 200
_batch_size = 32
_learning_rate = 0.01
_penalty = Const.L2
_reg_lambda = 0.01
_crypto_type = Const.PLAIN
_random_state = None
_key_size = 1024


# Load dataset
active_trainset = NDArrayDataset(role=Const.ACTIVE_NAME, abs_path=trainset_path)
active_testset = NDArrayDataset(role=Const.ACTIVE_NAME, abs_path=testset_path)


# Initialize messenger
_messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                               role=Const.ACTIVE_NAME,
                               active_ip=active_ip,
                               active_port=active_port,
                               passive_ip=passive_ip,
                               passive_port=passive_port)

# Dataset preprocessing
active_trainset = scale(add_intercept(active_trainset))
active_testset = scale(add_intercept(active_testset))


