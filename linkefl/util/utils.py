import os
import pathlib

import numpy as np
import torch

from linkefl.common.const import Const
from linkefl.config import LinearConfig
from linkefl.config import NNConfig
from linkefl.vfl.nn.model import (
    PassiveBottomModel, ActiveBottomModel,
    IntersectionModel, TopModel
)


def sigmoid(x):
    # return np.exp(x) / (1 + np.exp(x))
    return 1.0 / (1.0 + np.exp(-x))


def save_params(params, role):
    params_dir = './checkpoint/{}/{}/{}/'.format(LinearConfig.DATASET_NAME,
                                                LinearConfig.FEAT_SELECT_METHOD,
                                                LinearConfig.ATTACKER_FEATURES_FRAC)
    file_name = '{}_params.npy'.format(role)

    if not os.path.exists(params_dir):
        pathlib.Path(params_dir).mkdir(parents=True, exist_ok=True)
    with open(params_dir + file_name, 'wb') as f:
        np.save(f, params)


def load_params(role):
    params_dir = './checkpoint/{}/{}/{}/'.format(LinearConfig.DATASET_NAME,
                                                LinearConfig.FEAT_SELECT_METHOD,
                                                LinearConfig.ATTACKER_FEATURES_FRAC)
    file_name = '{}_params.npy'.format(role)

    with open(params_dir + file_name, 'rb') as f:
        return np.load(f)


def save_model(model, optimizer, epoch, model_name):
    """Save trained models to disk.

    Args:
        model: PyTorch model.
        optimizer: The optimizer associated with the model.
        epoch: Which training epoch the model is saved.
        model_name: Name of the model.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    model_dir = './checkpoint/{}/{}/{}'.format(NNConfig.DATASET_NAME,
                                               NNConfig.ATTACKER_FEATURES_FRAC,
                                               NNConfig.FEAT_SELECT_METHOD)
    # reference: https://stackoverflow.com/a/600612/8418540
    # create directories recursively, the same as `mkdir -p`
    if not os.path.exists(model_dir):
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, model_dir + '/{}.pth'.format(model_name))


def load_model(model_name):
    """Load model by its name.

    Args:
        model_name: Name of the model.

    Returns:
        PyTorch model.
    """
    if model_name == 'alice_bottom_model':
        model = PassiveBottomModel(NNConfig.ALICE_BOTTOM_NODES)

    elif model_name == 'bob_bottom_model':
        model = ActiveBottomModel(NNConfig.BOB_BOTTOM_NODES)

    elif model_name == 'intersection_model':
        model = IntersectionModel(NNConfig.INTERSECTION_NODES)

    elif model_name == 'top_model':
        model = TopModel(NNConfig.TOP_NODES)

    elif model_name == 'local_bottom':
        model = PassiveBottomModel(NNConfig.ALICE_BOTTOM_NODES)

    elif model_name == 'local_append':
        model = TopModel(NNConfig.APPEND_NODES)

    else:
        raise ValueError('Invalid model name, please check in again.')

    model_dir = './checkpoint/{}/{}/{}'.format(NNConfig.DATASET_NAME,
                                               NNConfig.ATTACKER_FEATURES_FRAC,
                                               NNConfig.FEAT_SELECT_METHOD)
    checkpoint = torch.load(model_dir + '/{}.pth'.format(model_name))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def save_data(data, name):
    """Save tensor to disk.

    Args:
        data: PyTorch tensor.
        name: Tensor name.
    """
    path = './checkpoint/{}/{}/{}/{}.pth'.format(NNConfig.DATASET_NAME,
                                                 NNConfig.ATTACKER_FEATURES_FRAC,
                                                 NNConfig.FEAT_SELECT_METHOD,
                                                 name)
    torch.save(data, path)


def load_data(name):
    """Loads tensor by name.

    Args:
        name: Tensor name.

    Returns:
        Loaded PyTorch tensor.
    """
    path = './checkpoint/{}/{}/{}/{}.pth'.format(NNConfig.DATASET_NAME,
                                                 NNConfig.ATTACKER_FEATURES_FRAC,
                                                 NNConfig.FEAT_SELECT_METHOD,
                                                 name)

    return torch.load(path)


def num_input_nodes(dataset_name, role, passive_feat_frac):
    assert dataset_name in Const.BUILDIN_DATASET, 'dataset_name should be one of' \
                                                  'the build-in datasets'
    assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'Invalid role'
    assert 0 < passive_feat_frac < 1, 'passive_feat_frac should be in range (0,1)'

    total_feats = {
        'cancer': 30,
        'digits': 64,
        'epsilon': 100,
        'census': 81,
        'credit': 10,
        'mnist': 28 * 28,
        'fashion_mnist': 28 * 28,
    }

    if role == Const.PASSIVE_NAME:
        num_nodes = int(total_feats[dataset_name] * passive_feat_frac)
    else:
        num_nodes = total_feats[dataset_name] - int(total_feats[dataset_name] * passive_feat_frac)

    return num_nodes


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params