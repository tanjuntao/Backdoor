import torch.optim.optimizer
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import TorchDataset
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import RSAPSIPassive
from linkefl.vfl.nn import PassiveBottomModel, PassiveNeuralNetwork


if __name__ == '__main__':
    # 0. Set parameters
    trainset_path = '/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/give-me-some-credit-passive-train.csv'
    testset_path = '/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/give-me-some-credit-passive-test.csv'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 30000
    _epochs = 80
    _batch_size = 64
    _learning_rate = 0.01
    _crypto_type = Const.PLAIN
    _key_size = 1024
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    bottom_nodes = [5, 3, 3]

    # 1. Load dataset
    passive_trainset = TorchDataset.from_csv(
        role=Const.PASSIVE_NAME,
        abs_path=trainset_path,
        dataset_type=Const.CLASSIFICATION
    )
    passive_testset = TorchDataset.from_csv(
        role=Const.PASSIVE_NAME,
        abs_path=testset_path,
        dataset_type=Const.CLASSIFICATION
    )
    print(colored('1. Finish loading dataset.', 'red'))

    # # 2. Feature transformation
    # passive_trainset = scale(passive_trainset)
    # passive_testset = scale(passive_testset)
    # print(colored('2. Finish transforming features', 'red'))

    # 3. Run PSI
    print(colored('3. PSI protocol started, computing...', 'red'))
    messenger = FastSocket(role=Const.PASSIVE_NAME,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port)
    passive_psi = RSAPSIPassive(passive_trainset.ids, messenger, _logger)
    common_ids = passive_psi.run()
    passive_trainset.filter(common_ids)
    print(colored('3. Finish psi protocol', 'red'))

    # 4. VFL training
    bottom_model = PassiveBottomModel(bottom_nodes)
    optimizer = torch.optim.SGD(bottom_model.parameters(), lr=_learning_rate)
    passive_vfl = PassiveNeuralNetwork(epochs=_epochs,
                                       batch_size=_batch_size,
                                       model=bottom_model,
                                       optimizer=optimizer,
                                       messenger=messenger,
                                       crypto_type=_crypto_type,
                                       saving_model=True)
    passive_vfl.train(passive_trainset, passive_testset)
    print(colored('4. Finish collaborative model training', 'red'))

    # 5. inference
    scores = passive_vfl.validate(passive_testset)
    print(scores)

    # 6. Finish the whole pipeline
    messenger.close()
    print(colored('All Done.', 'red'))

    # For online inference, you just need to substitute the model_name
    # scores = PassiveNeuralNetwork.online_inference(
    #     passive_testset,
    #     messenger,
    #     model_arch=bottom_model,
    #     model_name='20220901_174607-passive_party-vertical_nn-120000_samples.model',
    # )
    # print(scores)