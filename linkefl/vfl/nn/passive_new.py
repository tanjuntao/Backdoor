import datetime
import time

from termcolor import colored
import torch
from torch.utils.data import DataLoader

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.dataio import TorchDataset
from linkefl.util import num_input_nodes
from linkefl.vfl.nn.enc_layer import PassiveEncLayer
from linkefl.vfl.nn.model import MLPModel, CutLayer


class PassiveNeuralNetwork:
    def __init__(self,
                 epochs,
                 batch_size,
                 learning_rate,
                 models,
                 optimizers,
                 messenger,
                 cryptosystem,
                 *,
                 precision=0.001,
                 random_state=None,
                 saving_model=False,
                 model_path='./models',
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = models
        self.optimizers = optimizers
        self.messenger = messenger
        self.cryptosystem = cryptosystem
        if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
            self.enc_layer = PassiveEncLayer(
                in_nodes=self.models["cut"].in_nodes,
                out_nodes=self.models["cut"].out_nodes,
                eta=learning_rate,
                messenger=messenger,
                cryptosystem=cryptosystem
            )
        self._sync_pubkey()

        self.precision = precision
        self.random_state = random_state
        self.saving_model = saving_model
        self.model_path = model_path
        self.model_name = "{time}-{role}-{model_type}".format(
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            role=Const.PASSIVE_NAME,
            model_type=Const.VERTICAL_NN
        )

    def _sync_pubkey(self):
        print('Training protocol started.')
        print('Sending public key...')
        self.messenger.send(self.cryptosystem.pub_key)
        print('Done.')

    def _init_dataloader(self, dataset, shuffle=False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self, trainset, testset):
        assert isinstance(trainset, TorchDataset), \
            'trainset should be an instance of TorchDataset'
        assert isinstance(testset, TorchDataset), \
            'testset should be an instance of TorchDataset'
        train_dataloader = self._init_dataloader(trainset)
        test_dataloader = self._init_dataloader(testset)

        for model in self.models.values():
            model.train()

        start_time = time.time()
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch))
            for batch_idx, X in enumerate(train_dataloader):
                print(f"batch: {batch_idx}")
                # 1. forward
                bottom_outputs = self.models["bottom"](X)
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    # TODO: whether using bottom_outputs.data
                    passive_repr = self.enc_layer.fed_forward(bottom_outputs)
                else:
                    passive_repr = self.models["cut"](bottom_outputs)
                self.messenger.send(passive_repr.data)

                # 2. backward
                for optmizer in self.optimizers.values():
                    optmizer.zero_grad()
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    self.enc_layer.fed_backward()
                    grad = self.messenger.recv()
                    plain_grad = self.cryptosystem.decrypt_vector(grad.flatten())
                    plain_grad = torch.tensor(plain_grad).reshape(grad.shape)
                    bottom_outputs.backward(plain_grad)
                    self.optimizers["bottom"].step()
                else:
                    grad = self.messenger.recv()
                    passive_repr.backward(grad)
                    self.optimizers["cut"].step()
                    self.optimizers["bottom"].step()

                break

            scores = self.validate(testset, existing_loader=test_dataloader)
            is_best = self.messenger.recv()
            if is_best:
                print(colored('Best model updated.', 'red'))

        print(colored('Total training and validation time: {}'
                      .format(time.time() - start_time), 'red'))

    def validate(self, testset, existing_loader=None):
        if existing_loader is None:
            dataloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        else:
            dataloader = existing_loader

        for model in self.models.values():
            model.eval()

        with torch.no_grad():
            for batch, X in enumerate(dataloader):
                bottom_outputs = self.models["bottom"](X)
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    passive_repr = self.enc_layer.fed_forward(bottom_outputs)
                else:
                    passive_repr = self.models["cut"](bottom_outputs)
                self.messenger.send(passive_repr)
            scores = self.messenger.recv()
            return scores


if __name__ == '__main__':
    # 0. Set parameters
    dataset_name = 'fashion_mnist'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 30000
    _epochs = 2
    _batch_size = 100
    _learning_rate = 0.01
    _crypto_type = Const.FAST_PAILLIER
    _key_size = 1024
    torch.manual_seed(1314)

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
    print('Done.')

    # 2. Create PyTorch models and optimizers
    input_nodes = num_input_nodes(
        dataset_name=dataset_name,
        role=Const.PASSIVE_NAME,
        passive_feat_frac=passive_feat_frac
    )
    bottom_nodes = [input_nodes, 256, 128]
    cut_nodes = [128, 64]
    bottom_model = MLPModel(bottom_nodes, activate_input=False, activate_output=True)
    cut_layer = CutLayer(*cut_nodes)
    _models = {"bottom": bottom_model, "cut": cut_layer}
    _optimizers = {name: torch.optim.SGD(model.parameters(), lr=_learning_rate)
                   for name, model in _models.items()}

    # 3. Initialize messenger and cryptosystem
    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.PASSIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)
    _crypto = crypto_factory(crypto_type=_crypto_type,
                             key_size=_key_size,
                             num_enc_zeros=100,
                             gen_from_set=False)

    # 4. Initialize vertical NN protocol and start fed training
    passive_party = PassiveNeuralNetwork(epochs=_epochs,
                                         batch_size=_batch_size,
                                         learning_rate=_learning_rate,
                                         models=_models,
                                         optimizers=_optimizers,
                                         messenger=_messenger,
                                         cryptosystem=_crypto,)
    passive_party.train(passive_trainset, passive_testset)

    # 5. Close messenger, finish training
    _messenger.close()