from termcolor import colored
import torch
from torch import nn

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory
from linkefl.crypto import RSACrypto
from linkefl.dataio import TorchDataset
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import RSAPSIActive
from linkefl.vfl.nn import ActiveBottomModel, IntersectionModel, TopModel
from linkefl.vfl.nn import ActiveNeuralNetwork



if __name__ == '__main__':
    # 0. Set parameters
    trainset_path = '/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/give-me-some-credit-active-train.csv'
    testset_path = '/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/give-me-some-credit-active-test.csv'
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
    _loss_fn = nn.CrossEntropyLoss()
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    bottom_nodes = [5, 3, 3]
    intersect_nodes = [3, 3, 3]
    top_nodes = [3, 2]

    # 1. Load dataset
    active_trainset = TorchDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path=trainset_path,
        dataset_type=Const.CLASSIFICATION
    )
    active_testset = TorchDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path=testset_path,
        dataset_type=Const.CLASSIFICATION
    )
    print(colored('1. Finish loading dataset.', 'red'))

    # # 2. Feature transformation
    # active_trainset = scale(add_intercept(active_trainset))
    # active_testset = scale(add_intercept(active_testset))
    # print(colored('2. Finish transforming features', 'red'))

    # 3. Run PSI
    print(colored('3. PSI protocol started, computing...', 'red'))
    messenger = FastSocket(role=Const.ACTIVE_NAME,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port)
    psi_crypto = RSACrypto()
    active_psi = RSAPSIActive(active_trainset.ids, messenger, psi_crypto, _logger)
    common_ids = active_psi.run()
    active_trainset.filter(common_ids)
    print(colored('3. Finish psi protocol', 'red'))

    # 4. VFL training
    bottom_model = ActiveBottomModel(bottom_nodes)
    intersect_model = IntersectionModel(intersect_nodes)
    top_model = TopModel(top_nodes)
    _models = [bottom_model, intersect_model, top_model]
    _optimizers = [torch.optim.SGD(model.parameters(), lr=_learning_rate)
                   for model in _models]
    vfl_crypto = crypto_factory(crypto_type=_crypto_type,
                                key_size=_key_size,
                                num_enc_zeros=10000,
                                gen_from_set=False)
    active_vfl = ActiveNeuralNetwork(epochs=_epochs,
                                     batch_size=_batch_size,
                                     models=_models,
                                     optimizers=_optimizers,
                                     loss_fn=_loss_fn,
                                     messenger=messenger,
                                     crypto_type=_crypto_type,
                                     saving_model=True)
    active_vfl.train(active_trainset, active_testset)

    # 5. inference
    scores = active_vfl.validate(active_testset)
    print(scores)

    # 6. Close messenger, finish training
    messenger.close()


    # For online inference, you only need to substitute the model name
    # scores = ActiveNeuralNetwork.online_inference(
    #     active_testset,
    #     messenger,
    #     model_arch=_models,
    #     model_name='20220901_174600-active_party-vertical_nn-120000_samples.model',
    # )
    # print(scores)