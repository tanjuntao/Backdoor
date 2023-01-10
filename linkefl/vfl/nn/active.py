import datetime
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from termcolor import colored
from torch import nn
from torch.utils.data import DataLoader

from linkefl.common.const import Const
from linkefl.dataio import TorchDataset
from linkefl.modelio import TorchModelIO


class ActiveNeuralNetwork:
    def __init__(self,
                 epochs,
                 batch_size,
                 models,
                 optimizers,
                 loss_fn,
                 messenger,
                 crypto_type,
                 logger,
                 *,
                 precision=0.001,
                 random_state=None,
                 saving_model=False,
                 model_path='./models',
                 model_name=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = models
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.messenger = messenger
        self.crypto_type = crypto_type
        self.logger = logger

        self.precision = precision
        self.random_state = random_state
        self.saving_model = saving_model
        self.model_path = model_path
        if model_name is None:
            self.model_name = "{time}-{role}-{model_type}".format(
                time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                role=Const.ACTIVE_NAME,
                model_type=Const.VERTICAL_NN
            )
        else:
            self.model_name = model_name

    def _init_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def train(self, trainset, testset):
        assert isinstance(trainset, TorchDataset), 'trainset should be an instance ' \
                                                   'of TorchDataset'
        assert isinstance(testset, TorchDataset), 'testset should be an instance ' \
                                                  'of TorchDataset'
        train_dataloader = self._init_dataloader(trainset)
        test_dataloader = self._init_dataloader(testset)

        for model in self.models:
            model.train()
        start_time = None
        best_acc, best_auc = 0, 0
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch + 1))
            for batch_idx, (X, y) in enumerate(train_dataloader):
                passive_data = self.messenger.recv()
                if start_time is None:
                    start_time = time.time()
                _, passive_size = passive_data.shape
                active_bottom = self.models[0](X)
                concat = torch.cat((passive_data, active_bottom.data), dim=1)
                concat = concat.requires_grad_()

                outputs = self.models[2](self.models[1](concat))
                loss = self.loss_fn(outputs, y)
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                loss.backward()
                self.optimizers[1].step()
                self.optimizers[2].step()

                self.messenger.send(concat.grad[:, :passive_size])
                active_bottom.backward(concat.grad[:, passive_size:])
                self.optimizers[0].step()

                if batch_idx % 100 == 0:
                    loss, current = loss.item(), batch_idx * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{trainset.n_samples:>5d}]")

            is_best = False
            scores = self.validate(testset, existing_loader=test_dataloader)
            curr_acc, curr_auc, curr_loss = scores['acc'], scores['auc'], scores['loss']
            self.logger.log_metric(epoch,
                                   curr_loss,
                                   scores['acc'], scores['auc'], 0,
                                   total_epoch=self.epochs)
            if curr_acc > best_acc:
                print(colored('Best model update.\n', 'red'))
                is_best = True
                best_acc = curr_acc
                if self.saving_model:
                    # model_name = self.model_name + "-" + str(trainset.n_samples) + "_samples" + ".model"
                    model_name = self.model_name
                    TorchModelIO.save(self.models,
                                      self.model_path,
                                      model_name,
                                      epoch=epoch,
                                      optimizer=self.optimizers)
            if curr_auc > best_auc:
                best_auc = curr_auc
            self.messenger.send(is_best)

        print(colored('Total training and validation time: {:.4f}'
                      .format(time.time() - start_time), 'red'))
        print(colored('Best testing accuracy: {:.5f}'.format(best_acc), 'red'))
        print(colored('Best testing auc: {:.5f}'.format(best_auc), 'red'))

    def validate(self, testset, existing_loader=None):
        scores = ActiveNeuralNetwork._validate_util(
            testset, self.messenger,
            model=self.models,
            existing_loader=existing_loader,
            loss_fn=self.loss_fn
        )
        return scores
        # assert isinstance(testset, TorchDataset), 'testset should be an instance ' \
        #                                           'of TorchDataset'
        # if existing_loader is None:
        #     test_dataloader = self._init_dataloader(testset)
        # else:
        #     test_dataloader = existing_loader
        # for model in self.models:
        #     model.eval()
        #
        # num_batches = len(test_dataloader)
        # test_loss = 0
        # correct = 0
        # labels, probs = np.array([]), np.array([]) # used for computing AUC score
        # with torch.no_grad():
        #     for batch, (X, y) in enumerate(test_dataloader):
        #         passive_data = self.messenger.recv()
        #         active_bottom = self.models[0](X)
        #         concat = torch.cat((passive_data, active_bottom.data), dim=1)
        #         outputs = self.models[2](self.models[1](concat))
        #         self.messenger.send(outputs)
        #
        #         labels = np.append(labels, y.numpy().astype(np.int32))
        #         probs = np.append(probs, torch.sigmoid(outputs[:, 1]).numpy())
        #
        #         test_loss += self.loss_fn(outputs, y).item()
        #         correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
        #
        #     test_loss /= num_batches
        #     acc = correct / testset.n_samples
        #     n_classes = len(torch.unique(testset.labels))
        #     if n_classes == 2:
        #         auc = roc_auc_score(labels, probs)
        #     else:
        #         auc = 0
        #     print(f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%,"
        #           f" Auc: {(100 * auc):>0.2f}%,"
        #           f" Avg loss: {test_loss:>8f}")
        #
        #     return acc, auc

    @staticmethod
    def online_inference(dataset, messenger,
                         model_arch, model_name, model_path='./models',
                         optimizer_arch=None,
                         infer_step=64, loss_fn=None):
        scores = ActiveNeuralNetwork._validate_util(
            dataset,
            messenger,
            model_arch=model_arch,
            model_path=model_path,
            model_name=model_name,
            optimizer_arch=optimizer_arch,
            infer_step=infer_step,
            loss_fn=loss_fn
        )
        return scores

    @staticmethod
    def _validate_util(dataset, messenger, *,
                       model=None,
                       model_arch=None, model_path='./models', model_name=None,
                       optimizer_arch=None,
                       existing_loader=None, infer_step=64, loss_fn=None):
        assert isinstance(dataset, TorchDataset), 'dataset should be an instance' \
                                                  'of TorchDataset'
        if model is None:
            model, _, _ = TorchModelIO.load(model_arch, model_path, model_name,
                                            optimizer_arch=optimizer_arch)
        if existing_loader is None:
            dataloader = DataLoader(dataset, batch_size=infer_step, shuffle=False)
        else:
            dataloader = existing_loader
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        for sub_model in model:
            sub_model.eval()
        num_batches = len(dataloader)
        test_loss = 0
        correct = 0
        labels, probs = np.array([]), np.array([])  # used for computing AUC score
        preds = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                passive_data = messenger.recv()
                active_bottom = model[0](X)
                concat = torch.cat((passive_data, active_bottom.data), dim=1)
                outputs = model[2](model[1](concat))
                # messenger.send(outputs)

                labels = np.append(labels, y.numpy().astype(np.int32))
                probs = np.append(probs, torch.sigmoid(outputs[:, 1]).numpy())

                test_loss += loss_fn(outputs, y).item()
                correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
                preds.extend(outputs.argmax(1).numpy().tolist())

            test_loss /= num_batches
            acc = correct / dataset.n_samples
            n_classes = len(torch.unique(dataset.labels))
            if n_classes == 2:
                auc = roc_auc_score(labels, probs)
            else:
                auc = 0
            print(f"Test Error: \n Accuracy: {(100 * acc):>0.2f}%,"
                  f" Auc: {(100 * auc):>0.2f}%,"
                  f" Avg loss: {test_loss:>8f}")

            scores = {
                "acc": acc,
                "auc": auc,
                "loss": test_loss,
                "preds": preds
            }
            messenger.send(scores)
            return scores


if __name__ == '__main__':
    from linkefl.common.factory import messenger_factory, logger_factory
    from linkefl.util import num_input_nodes
    from linkefl.vfl.nn.model import ActiveBottomModel, IntersectionModel, TopModel

    # 0. Set parameters
    dataset_name = 'tab_mnist'
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
    _loss_fn = nn.CrossEntropyLoss()

    # 1. Load datasets
    print('Loading dataset...')
    active_trainset = TorchDataset.buildin_dataset(dataset_name=dataset_name,
                                                   role=Const.ACTIVE_NAME,
                                                   root='../data',
                                                   train=True,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    active_testset = TorchDataset.buildin_dataset(dataset_name=dataset_name,
                                                  role=Const.ACTIVE_NAME,
                                                  root='../data',
                                                  train=False,
                                                  download=True,
                                                  passive_feat_frac=passive_feat_frac,
                                                  feat_perm_option=feat_perm_option)
    # active_trainset = BuildinTorchDataset(dataset_name=dataset_name,
    #                                       role=Const.ACTIVE_NAME,
    #                                       train=True,
    #                                       passive_feat_frac=passive_feat_frac,
    #                                       feat_perm_option=feat_perm_option)
    # active_testset = BuildinTorchDataset(dataset_name=dataset_name,
    #                                      role=Const.ACTIVE_NAME,
    #                                      train=False,
    #                                      passive_feat_frac=passive_feat_frac,
    #                                      feat_perm_option=feat_perm_option)
    print('Done.')
    # for epsilon dataset, scale() must be applied.
    # active_trainset = scale(active_trainset)
    # active_testset = scale(active_testset)

    # 2. Created PyTorch models and associated optimizers
    input_nodes = num_input_nodes(dataset_name=dataset_name,
                                  role=Const.ACTIVE_NAME,
                                  passive_feat_frac=passive_feat_frac)
    # mnist & fashion_mnist
    bottom_nodes = [input_nodes, 256, 128]
    intersect_nodes = [128, 128, 10]
    top_nodes = [10, 10]

    # criteo
    # bottom_nodes = [input_nodes, 15, 10]
    # intersect_nodes = [10, 10, 10]
    # top_nodes = [10, 2]

    # avazu
    # bottom_nodes = [input_nodes, 10, 5]
    # intersect_nodes = [5, 5, 5]
    # top_nodes = [5, 2]

    # census
    # bottom_nodes = [input_nodes, 20, 10]
    # intersect_nodes = [10, 10, 10]
    # top_nodes = [10, 2]

    # credit
    # bottom_nodes = [input_nodes, 3, 3]
    # intersect_nodes = [3, 3, 3]
    # top_nodes = [3, 2]

    # default_credit
    # bottom_nodes = [input_nodes, 8, 5]
    # intersect_nodes = [5, 5, 5]
    # top_nodes = [5, 2]

    # epsilon
    # bottom_nodes = [input_nodes, 25, 10]
    # intersect_nodes = [10, 10, 10]
    # top_nodes = [10, 2]

    bottom_model = ActiveBottomModel(bottom_nodes)
    intersect_model = IntersectionModel(intersect_nodes)
    top_model = TopModel(top_nodes)
    _models = [bottom_model, intersect_model, top_model]
    _optimizers = [torch.optim.SGD(model.parameters(), lr=_learning_rate)
                   for model in _models]

    # 3. Initialize messenger
    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.ACTIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)
    _logger = logger_factory(role=Const.ACTIVE_NAME,
                             writing_file=False,
                             writing_http=False)
    print('Active party started, listening...')

    # 4. Initialize NN protocol and start training
    active_party = ActiveNeuralNetwork(epochs=_epochs,
                                       batch_size=_batch_size,
                                       models=_models,
                                       optimizers=_optimizers,
                                       loss_fn=_loss_fn,
                                       messenger=_messenger,
                                       crypto_type=_crypto_type,
                                       logger=_logger,
                                       saving_model=True,)
    active_party.train(active_trainset, active_testset)

    # 5. Close messenger, finish training
    _messenger.close()


    # # For online inference
    # _scores = ActiveNeuralNetwork.online_inference(
    #     active_testset,
    #     _messenger,
    #     model_arch=_models,
    #     model_name='20220901_120137-active_party-vertical_nn-60000_samples.model',
    #     optimizer_arch=_optimizers,
    # )
    # print(_scores)


