import numpy as np
from sklearn import preprocessing

from linkefl.common.const import Const
from linkefl.dataio import BaseDataset


def scale(ndarray_dataset):
    assert isinstance(ndarray_dataset, BaseDataset), \
        'dataset type must be an instance of BaseDataset'

    scaled_feats = preprocessing.scale(ndarray_dataset.features)
    ids = ndarray_dataset.ids
    try:
        labels = ndarray_dataset.labels
    except AttributeError:
        labels = None

    if labels is None:
        new_dataset = np.concatenate((ids[:, np.newaxis], scaled_feats), axis=1)
    else:
        new_dataset = np.concatenate((ids[:, np.newaxis],
                                      labels[:, np.newaxis],
                                      scaled_feats), axis=1)

    ndarray_dataset.set_dataset(new_dataset)
    return ndarray_dataset


def normalize(ndarray_dataset, norm=Const.L2):
    assert isinstance(ndarray_dataset, BaseDataset), \
        'dataset type must be an instance of BaseDataset'
    assert norm in (Const.L1, Const.L2), 'norm not supported at the moment'

    normalized_feats = preprocessing.normalize(ndarray_dataset.features, norm=norm)
    ids = ndarray_dataset.ids
    try:
        labels = ndarray_dataset.labels
    except AttributeError:
        labels = None

    if labels is None:
        new_dataset = np.concatenate((ids[:, np.newaxis], normalized_feats), axis=1)
    else:
        new_dataset = np.concatenate((ids[:, np.newaxis],
                                      labels[:, np.newaxis],
                                      normalized_feats), axis=1)

    ndarray_dataset.set_dataset(new_dataset)
    return ndarray_dataset


def add_intercept(ndarray_dataset):
    assert isinstance(ndarray_dataset, BaseDataset), \
        'dataset type must be an instance of BaseDataset'

    n_samples = ndarray_dataset.n_samples
    new_feats = np.c_[ndarray_dataset.features, np.zeros(n_samples)]
    ids = ndarray_dataset.ids
    try:
        labels = ndarray_dataset.labels
    except AttributeError:
        labels = None

    if labels is None:
        new_dataset = np.concatenate((ids[:, np.newaxis], new_feats,), axis=1)
    else:
        # convert negative label from -1 to 0
        labels[labels == -1] = 0
        new_dataset = np.concatenate((ids[:, np.newaxis],
                                      labels[:, np.newaxis],
                                      new_feats), axis=1)

    ndarray_dataset.set_dataset(new_dataset)
    return ndarray_dataset