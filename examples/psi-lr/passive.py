from termcolor import colored

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.feature import scale
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import RSAPSIPassive
from linkefl.vfl.linear import PassiveLogReg


if __name__ == '__main__':
    # 0. Set parameters
    trainset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_passive_train.csv'
    testset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_passive_test.csv'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 30000
    _epochs = 200
    _batch_size = 100
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.001
    _crypto_type = Const.PLAIN
    _random_state = None

    # 1. Load dataset
    passive_trainset = NumpyDataset(role=Const.PASSIVE_NAME, abs_path=trainset_path)
    passive_testset = NumpyDataset(role=Const.PASSIVE_NAME, abs_path=testset_path)
    print(colored('1. Finish loading dataset.', 'red'))

    # 2. Feature transformation
    passive_trainset = scale(passive_trainset)
    passive_testset = scale(passive_testset)
    print(colored('2. Finish transforming features', 'red'))

    # 3. Run PSI
    messenger = FastSocket(role=Const.PASSIVE_NAME,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port)
    passive_psi = RSAPSIPassive(passive_trainset.ids, messenger)
    common_ids = passive_psi.run()
    passive_trainset.filter(common_ids)
    print(colored('3. Finish psi protocol', 'red'))

    # 4. VFL training
    passive_vfl = PassiveLogReg(epochs=_epochs,
                                batch_size=_batch_size,
                                learning_rate=_learning_rate,
                                messenger=messenger,
                                crypto_type=_crypto_type,
                                penalty=_penalty,
                                reg_lambda=_reg_lambda,
                                random_state=_random_state,
                                using_pool=False)
    passive_vfl.train(passive_trainset, passive_testset)
    print(colored('4. Finish collaborative model training', 'red'))

    # VFL inference
    passive_vfl.predict(passive_testset)
    print(colored('5. Finish collaborative inference', 'red'))

    # 6. Finish the whole pipeline
    messenger.close()
    print(colored('All Done.', 'red'))