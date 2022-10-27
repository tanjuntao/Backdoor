import datetime
import time

from termcolor import colored
import torch
from torch.utils.data import DataLoader

from linkefl.common.const import Const
from linkefl.common.factory import messenger_factory,messenger_factory_v1
from linkefl.dataio import TorchDataset
from linkefl.feature.transform import scale
from linkefl.modelio import TorchModelIO
from linkefl.util import num_input_nodes
from linkefl.vfl.nn.model import PassiveBottomModel


class PassiveNeuralNetwork:
    def __init__(self,
                 epochs,
                 batch_size,
                 model,
                 optimizer,
                 messenger,
                 crypto_type,
                 *,
                 precision=0.001,
                 random_state=None,
                 saving_model=False,
                 model_path='./models',
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.messenger = messenger
        self.crypto_type = crypto_type

        self.precision = precision
        self.random_state = random_state
        self.saving_model = saving_model
        self.model_path = model_path
        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.PASSIVE_NAME,
            model_type=Const.VERTICAL_NN
        )

    def _init_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def train(self, trainset, testset):
        assert isinstance(trainset, TorchDataset), 'trainset should be an instance ' \
                                                   'of TorchDataset'
        assert isinstance(testset, TorchDataset), 'testset should be an instance' \
                                                  'of TorchDataset'
        train_dataloader = self._init_dataloader(trainset)
        test_dataloader = self._init_dataloader(testset)

        self.model.train()
        start_time = time.time()
        for epoch in range(self.epochs):
            for batch_idx, X in enumerate(train_dataloader):
                outputs = self.model(X)
                self.messenger.send(outputs.data)

                grads = self.messenger.recv() # blocking
                self.optimizer.zero_grad()
                outputs.backward(grads)
                self.optimizer.step()

            self.validate(testset, existing_loader=test_dataloader)
            is_best = self.messenger.recv()
            if is_best:
                print(colored('Best model updated.', 'red'))
                if self.saving_model:
                    model_name = self.model_name + "-" + str(trainset.n_samples) + "_samples" + ".model"
                    TorchModelIO.save(self.model,
                                      self.model_path,
                                      model_name,
                                      epoch=epoch,
                                      optimizer=self.optimizer)
            print(f"Epoch {epoch+1} finished.\n")
        print(colored('Total training and validation time: {:.2f}'
                      .format(time.time() - start_time), 'red'))

    def validate(self, testset, existing_loader=None):
        scores = PassiveNeuralNetwork._validate_util(
            testset, self.messenger,
            model=self.model,
            existing_loader=existing_loader
        )
        return scores

        # assert isinstance(testset, TorchDataset), 'testset should be an instance ' \
        #                                           'of TorchDataset'
        # if existing_loader is None:
        #     test_dataloader = self._init_dataloader(testset)
        # else:
        #     test_dataloader = existing_loader
        #
        # self.model.eval()
        # with torch.no_grad():
        #     for batch, X in enumerate(test_dataloader):
        #         outputs = self.model(X)
        #         self.messenger.send(outputs.data)
        #         pred = _messenger.recv() # pred for specific usage

    @staticmethod
    def online_inference(dataset, messenger,
                         model_arch, model_name, model_path='./models',
                         optimizer_arch=None,
                         infer_step=64):
        scores = PassiveNeuralNetwork._validate_util(
            dataset,
            messenger,
            model_arch=model_arch,
            model_path=model_path,
            model_name=model_name,
            optimizer_arch=optimizer_arch,
            infer_step=infer_step
        )
        return scores

    @staticmethod
    def _validate_util(dataset, messenger, *,
                       model=None,
                       model_arch=None, model_path='./models', model_name=None,
                       optimizer_arch=None,
                       existing_loader=None, infer_step=64):
        assert isinstance(dataset, TorchDataset), 'dataset should be an instance ' \
                                                  'of TorchDataset'
        if model is None:
            model, _, _ = TorchModelIO.load(model_arch, model_path, model_name,
                                            optimizer_arch=optimizer_arch)
        if existing_loader is None:
            dataloader = DataLoader(dataset, batch_size=infer_step, shuffle=False)
        else:
            dataloader = existing_loader

        model.eval()
        with torch.no_grad():
            for batch, X in enumerate(dataloader):
                outputs = model(X)
                messenger.send(outputs)
                # pred = messenger.recv()
        scores = messenger.recv()
        return scores


if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'census'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 30000
    _epochs = 100
    _batch_size = 200
    _learning_rate = 0.01
    _crypto_type = Const.PLAIN

    # 1. Load datasets
    print('Loading dataset...')
    passive_trainset = TorchDataset.buildin_dataset(dataset_name=dataset_name,
                                                    role=Const.PASSIVE_NAME,
                                                    root='../data',
                                                    train=True,
                                                    download=True,
                                                    passive_feat_frac=passive_feat_frac,
                                                    feat_perm_option=feat_perm_option)
    passive_testset = TorchDataset.buildin_dataset(dataset_name=dataset_name,
                                                   role=Const.PASSIVE_NAME,
                                                   root='../data',
                                                   train=False,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    # passive_trainset = BuildinTorchDataset(dataset_name=dataset_name,
    #                                        role=Const.PASSIVE_NAME,
    #                                        train=True,
    #                                        passive_feat_frac=passive_feat_frac,
    #                                        feat_perm_option=feat_perm_option)
    # passive_testset = BuildinTorchDataset(dataset_name=dataset_name,
    #                                       role=Const.PASSIVE_NAME,
    #                                       train=False,
    #                                       passive_feat_frac=passive_feat_frac,
    #                                       feat_perm_option=feat_perm_option)
    print('Done.')
    # for epsilon dataset, scale() must be applied.
    # passive_trainset = scale(passive_trainset)
    # passive_testset = scale(passive_testset)

    # 2. Create PyTorch model and optimizer
    input_nodes = num_input_nodes(dataset_name=dataset_name,
                                  role=Const.PASSIVE_NAME,
                                  passive_feat_frac=passive_feat_frac)
    # all_nodes = [input_nodes, 256, 128] # mnist & fashion_mnist
    # all_nodes = [input_nodes, 15, 10] # criteo
    # all_nodes = [input_nodes, 10, 5] # avazu
    all_nodes = [input_nodes, 20, 10] # census
    # all_nodes = [input_nodes, 3, 3] # credit
    # all_nodes = [input_nodes, 8, 5] # default_credit
    # all_nodes = [input_nodes, 25, 10] # epsilon
    passive_bottom_model = PassiveBottomModel(all_nodes)
    _optimizer = torch.optim.SGD(passive_bottom_model.parameters(), lr=_learning_rate)

    # 3. Initialize messenger
    _messenger = messenger_factory_v1(messenger_type=Const.FAST_SOCKET,
                                   role=Const.PASSIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)

    # 4. Initilize NN protocol and start training
    passive_party = PassiveNeuralNetwork(epochs=_epochs,
                                         batch_size=_batch_size,
                                         model=passive_bottom_model,
                                         optimizer=_optimizer,
                                         messenger=_messenger,
                                         crypto_type=_crypto_type,
                                         saving_model=False)
    passive_party.train(passive_trainset, passive_testset)

    # 5. Close messenger, finish training
    _messenger.close()


    # # For online inference
    # _scores = PassiveNeuralNetwork.online_inference(
    #     passive_testset,
    #     _messenger,
    #     model_arch=passive_bottom_model,
    #     model_name='20220901_120152-passive_party-vertical_nn-60000_samples.model',
    #     optimizer_arch=_optimizer
    # )
    # print(_scores)

