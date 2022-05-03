import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from termcolor import colored
from sklearn.metrics import roc_auc_score
import numpy as np

from linkefl.dataio import get_tensor_dataset as get_dataset
from linkefl.messenger import Socket, FastSocket
from linkefl.config import NNConfig as Config
from linkefl.util import save_model, save_data
from .model import BobBottomModel, IntersectionModel, TopModel


#############################
# Load np_dataset
#############################

training_data = get_dataset(dataset_name=Config.DATASET_NAME,
                            role='bob',
                            train=True,
                            attacker_features_frac=Config.ATTACKER_FEATURES_FRAC,
                            permutation=Config.PERMUTATION,
                            transform=ToTensor())
testing_data = get_dataset(dataset_name=Config.DATASET_NAME,
                           role='bob',
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
# MNIST and FashioMNIST
bottom_model = BobBottomModel(Config.BOB_BOTTOM_NODES).to(Config.BOB_DEVICE)
intersection_model = IntersectionModel(Config.INTERSECTION_NODES).to(Config.BOB_DEVICE)
top_model = TopModel(Config.TOP_NODES).to(Config.BOB_DEVICE)

# Census Income
# bottom_model = BobBottomModel([41, 20, 10])
# intersection_model = IntersectionModel(10, 10, 10)
# top_model = TopModel([10, 2])

# Give me some credit
# bottom_model = BobBottomModel([5, 3])
# intersection_model = IntersectionModel(3, 3, 6)
# top_model = TopModel([6, 3, 2])

# Sklearn digits
# bottom_model = BobBottomModel([32, 20, 16])
# intersection_model = IntersectionModel(16, 16, 16)
# top_model = TopModel([16, 10])
models = [bottom_model, intersection_model, top_model]
model_names = ['bob_bottom_model', 'intersection_model', 'top_model']
for model in models:
    print(model)

#############################
# Create optimizer
#############################
loss_fn = nn.CrossEntropyLoss()
optimizers = [torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE)
              for model in models]

def train(epoch, dataloader, models, optimizers, loss_fn, messenger):
    alice_reprs = None # RSAPSIPassive bottom model representations
    start_idx = 0

    if Config.TRAINING_VERBOSE:
        print('Epoch: {}'.format(epoch))
    size = len(dataloader.np_dataset)
    for model in models:
        model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(Config.BOB_DEVICE), y.to(Config.BOB_DEVICE)
        alice_data = messenger.recv() # shape: torch.Size([batch_size, output_size])

        if alice_reprs is None:
            alice_reprs = torch.zeros(size, alice_data.size(1))
        index = torch.arange(start_idx, start_idx + X.size(0))
        alice_reprs.index_copy_(0, index, alice_data)
        start_idx = start_idx + X.size(0)

        # Avoid using torch.cat which is really time-consuming
        # start = time.time()
        # if alice_reprs is None:
        #     alice_reprs = alice_data.clone()
        # else:
        #     alice_reprs = torch.cat((alice_reprs, alice_data), 0) # vertically
        #     pass

        alice_data = alice_data.to(Config.BOB_DEVICE)
        batch_size, alice_output_size = alice_data.shape
        bob_bottom = models[0](X)
        concat = torch.cat((alice_data, bob_bottom.data), 1) # horizontally
        concat = concat.requires_grad_()

        pred = models[2](models[1](concat))
        loss = loss_fn(pred, y)

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()

        for optimizer in optimizers[1:]:
            optimizer.step()

        # concat gard shape: torch.Size([batch_size, input_size of interseciton layer])
        messenger.send(concat.grad[:, :alice_output_size])
        bob_bottom.backward(concat.grad[:, alice_output_size:])
        optimizers[0].step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if Config.TRAINING_VERBOSE:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    training_reprs = {}
    n_classes = len(torch.unique(dataloader.np_dataset.targets))
    for label in range(n_classes):
        idxes = dataloader.np_dataset.targets == label
        reprs = alice_reprs[idxes]
        training_reprs[label] = reprs

    return training_reprs


def test(dataloader, models, loss_fn, messenger):
    size = len(dataloader.np_dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for model in models:
        model.eval()

    labels, probs = np.array([]), np.array([])
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(Config.BOB_DEVICE), y.to(Config.BOB_DEVICE)
            # if feature representation of passive_party bottom model is on GPU, then
            # tensor through pickle.loads() is still on GPU, we DON'T need to
            # transfer it to GPU manully.
            alice_data = messenger.recv()
            alice_data = alice_data.to(Config.BOB_DEVICE)
            bob_data = models[0](X)
            concat = torch.cat((alice_data, bob_data.data), 1)
            outputs = models[2](models[1](concat))
            messenger.send(outputs)

            labels = np.append(labels, y.numpy().astype(np.int32))
            probs = np.append(probs, torch.sigmoid(outputs[:, 1]).numpy())

            test_loss += loss_fn(outputs, y).item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size

        if Config.DATASET_NAME in ('mnist', 'fashion_mnist'):
            auc = 0
        else:
            auc = roc_auc_score(labels, probs)

        if Config.TRAINING_VERBOSE:
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.2f}%,"
                  f" Auc: {(100 * auc):>0.2f}%,"
                  f" Avg loss: {test_loss:>8f}")

        return correct, auc

#############################
# Driver
#############################
parser = argparse.ArgumentParser()
parser.add_argument('--retrain', action='store_true')
args = parser.parse_args()
if not args.retrain:
    exit(0)

messenger = FastSocket(role='bob', config=Config)
print('Listening...')

start = time.time()
best_test_acc, best_test_auc = 0.0, 0.0
for t in range(Config.EPOCHS):
    is_best = False
    training_reprs = train(t+1, train_dataloader, models, optimizers, loss_fn, messenger)
    test_acc, test_auc = test(test_dataloader, models, loss_fn, messenger)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        if Config.TRAINING_VERBOSE:
            print(colored('Best model saved.\n', 'red'))
        for model, optimizer, name in zip(models, optimizers, model_names):
            save_model(model, optimizer, t, name)
        save_data(data=training_reprs, name='training_reprs')
        is_best = True
    if test_auc > best_test_auc:
        best_test_auc = test_auc
    messenger.send(is_best)

messenger.close()
print('Total training and validation time: {:.2f}'.format(time.time() - start))
print('Best testing accuracy is: {}'.format(best_test_acc))
print('Best testing auc is: {}'.format(best_test_auc))
