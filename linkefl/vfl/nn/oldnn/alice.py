import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from termcolor import colored

from linkefl.common.const import Const
from linkefl.config import NNConfig as Config
from linkefl.dataio import get_tensor_dataset as get_dataset
from linkefl.messenger import FastSocket
from linkefl.vfl.nn.model import PassiveBottomModel
from linkefl.util import save_model


if __name__ == '__main__':
    #############################
    # Load np_dataset
    #############################
    training_data = get_dataset(dataset_name=Config.DATASET_NAME,
                                role=Const.PASSIVE_NAME,
                                train=True,
                                attacker_features_frac=Config.ATTACKER_FEATURES_FRAC,
                                permutation=Config.PERMUTATION,
                                transform=ToTensor())
    testing_data = get_dataset(dataset_name=Config.DATASET_NAME,
                               role=Const.PASSIVE_NAME,
                               train=False,
                               attacker_features_frac=Config.ATTACKER_FEATURES_FRAC,
                               permutation=Config.PERMUTATION,
                               transform=ToTensor())
    train_dataloader = DataLoader(training_data,
                                  batch_size=Config.BATCH_SIZE,
                                  shuffle=False)
    test_dataloader = DataLoader(testing_data,
                                 batch_size=Config.BATCH_SIZE,
                                 shuffle=False)

    #############################
    # Create model
    #############################
    # NUM_NODES = [int(28*28/2), 256, 128] # MNIST and FashioMNIST
    # NUM_NODES = [40, 20, 10] # Census Income
    # NUM_NODES = [32, 20, 16] # Sklearn digits
    # NUM_NODES = [5, 3] # Give me some credit
    bottom_model = PassiveBottomModel(Config.ALICE_BOTTOM_NODES).to(Config.ALICE_DEVICE)
    print(bottom_model)

    def train(dataloader, model, optimizer, messenger):
        model.train()
        for batch, X in enumerate(dataloader):
            X = X.to(Config.ALICE_DEVICE)
            outputs = model(X)
            # outputs = outputs.detach()
            messenger.send(outputs.data)

            grads = messenger.recv() # blocking
            grads = grads.to(Config.ALICE_DEVICE)
            optimizer.zero_grad()
            outputs.backward(grads)
            optimizer.step()

    def test(dataloader, model, messenger):
        model.eval()
        with torch.no_grad():
            for batch, X in enumerate(dataloader):
                X = X.to(Config.ALICE_DEVICE)
                outputs = model(X)
                messenger.send(outputs.data)

                pred = messenger.recv()

    #############################
    # Driver
    #############################
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    if not args.retrain:
        exit(0)


    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 30000


    messenger = FastSocket(role=Const.PASSIVE_NAME,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port)
    optimizer = torch.optim.SGD(bottom_model.parameters(), lr=Config.LEARNING_RATE)
    for t in range(Config.EPOCHS):
        train(train_dataloader, bottom_model, optimizer, messenger)
        test(test_dataloader, bottom_model, messenger)

        is_best = messenger.recv()
        if is_best:
            # save_model(bottom_model, optimizer, t, 'alice_bottom_model')
            print(colored('Best model saved.', 'red'))
        print('Epoch {} finished.\n'.format(t+1))

    messenger.close()


