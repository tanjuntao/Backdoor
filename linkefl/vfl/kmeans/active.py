import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset


class ActiveConstrainedSeedKMeans:
    """Constrained seed KMeans algorithm proposed by Basu et al. in 2002."""

    def __init__(self,
                 messenger,
                 crypto_type,
                 n_clusters=2,
                 *,
                 n_init=2,
                 max_iter=30,
                 tol=0.0001,
                 verbose=False,
                 invalide_label=-1,
                 random_state=None,
                 ):
        """Initialization a constrained seed kmeans estimator.
        Args:
            n_clusters: The number of clusters.
            n_init: The number of times the algorithm will run in order to choose
                the best result.
            max_iter: The maximum number of iterations the algorithm will run.
            tol: The convergence threshold of the algorithm. If the norm of a
                matrix, which is the difference_active between two consective cluster
                centers, is less than this threshold, we think the algorithm converges.
            verbose: Whether to print intermediate results to console.
            invalide_label: Special sign to indicate which samples are unlabeled.
                If the y value of a sample equals to this value, then that sample
                is a unlabeled one.
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.INVALID_LABEL = invalide_label
        self.messenger = messenger
        self.random_state = random_state
        if random_state is not None:
            torch.random.manual_seed(random_state)


    def _check_params(self, X_active, y):
        """Check if the parameters of the algorithm and the inputs to it are valid."""
        if type(X_active) not in (np.ndarray, torch.Tensor):
            raise TypeError(f"Type of X_active can only take numpy.ndarray and "
                            f"torch.Tensor, but got {type(X_active)} instead.")

        if type(y) not in (list, np.ndarray, torch.Tensor):
            raise TypeError(f"Type of y can only take list, numpy.ndarray, and"
                            f"torch.Tensor, but got{type(y)} instead.")

        if self.n_clusters > X_active.shape[0]:
            raise ValueError(f"The number of clusters mube be less than the "
                             f"number of samples.")

        if self.max_iter <= 0:
            raise ValueError(f"The number of maximum iteration must larger than zero.")

    def _init_centroids(self, X_active, y):
        """Initialize cluster centers with little samples having label."""
        if type(y) == np.ndarray:
            pkg = np
        elif type(y) == torch.Tensor:
            pkg = torch
        elif type(y) == list and type(X_active) == np.ndarray:
            y = np.array(y)
            pkg = np
        elif type(y) == list and type(X_active) == torch.Tensor:
            y = torch.Tensor(y)
            pkg = torch
        else:
            raise TypeError('Data type is not supported, please check it again.')

        y_unique = pkg.unique(y)
        if self.INVALID_LABEL in y_unique:
            n_seed_centroids = len(y_unique) - 1
        else:
            n_seed_centroids = len(y_unique)
        assert n_seed_centroids <= self.n_clusters, f"The number of seed centroids" \
                                                    f"should be less than the total" \
                                                    f"number of clusters."
        # n_seed_centroids 是利用打有标签的样本来确定的中心的个数


        self.messenger.send(n_seed_centroids)

        centers_active = pkg.empty((self.n_clusters, X_active.shape[1]), dtype=X_active.dtype)
        # First, initialize seed centers using samples with label
        labeled_idxes = []
        for i in range(n_seed_centroids):
            seed_samples = X_active[y == i]
            label_idx = pkg.where(y == i)[0]
            labeled_idxes.append(label_idx)
            centers_active[i] = seed_samples.mean(axis=0)
        
        print('labeled_idxes:', labeled_idxes)
        # print('labeled_idxes:', labeled_idxes[0])

        self.messenger.send(labeled_idxes)

        # Then, initilize the remaining centers with random samples from X_active
        unlabel_idxes = pkg.where(y == self.INVALID_LABEL)[0] # np.where returns a tuple
        self.messenger.send(unlabel_idxes)

        if len(unlabel_idxes) == 0:
            raise ValueError("All samples are labeled! No need for clustering!")

        if len(unlabel_idxes) < self.n_clusters - n_seed_centroids:
            np.random.seed(self.random_state)
            idx = np.random.randint(X_active.shape[0], size=self.n_clusters - n_seed_centroids)
            print('index', idx)

            for i in range(n_seed_centroids, self.n_clusters):
                centers_active[i] = X_active[idx[i - n_seed_centroids]]
        else:
            for i in range(n_seed_centroids, self.n_clusters):
                np.random.seed(self.random_state)
                idx = np.random.choice(unlabel_idxes, 1, replace=False)
                centers_active[i] = X_active[idx]

        print('centers_active', centers_active)

        return centers_active, n_seed_centroids



    def _kmeans(self, X_active, y, init_centers_active):
        """KMeans algorithm implementation."""
        indices = copy.copy(y)
        if type(indices) == list:
            indices = np.array(indices)
        n_samples, n_features = X_active.shape[0], X_active.shape[1]
        cur_centers = init_centers_active
        new_centers_active = copy.deepcopy(init_centers_active)

        valid_idxes = []

        for i in range(n_samples):
            if y[i] != self.INVALID_LABEL:
                valid_idxes.append(i)

        self.messenger.send(valid_idxes)
        print('invalid index', valid_idxes)

        # Main loop
        for iter_ in range(self.max_iter):
            # Fist step in KMeans: calculate the closest centroid for each sample
            for i in range(n_samples):
                # If this sample has label, then we use the ground-truth label
                # as its cluster index
                # if y[i] != self.INVALID_LABEL:
                #     continue
                if i in valid_idxes:
                    continue

                passive_norm = self.messenger.recv()

                if type(X_active) == np.ndarray:
                    min_idx = (np.square(np.linalg.norm(cur_centers - X_active[i], axis=1)) + np.square(passive_norm)).argmin()
                else:
                    min_idx = (torch.square(torch.norm(cur_centers - X_active[i], dim=1)) + torch.square(passive_norm)).argmin()

                print('min index', min_idx)
                indices[i] = min_idx

            self.messenger.send(indices)

            # Second step in KMeans: update each centroids
            for i in range(self.n_clusters):
                active_cluster_samples = X_active[indices == i]
                # In the case that the cluster is empty, randomly choose
                # a sample from X_active.
                if active_cluster_samples.shape[0] == 0:
                    np.random.seed(self.random_state)
                    new_centers_active[i] = X_active[np.random.choice(n_samples, 1, replace=False)]
                else:
                    new_centers_active[i] = active_cluster_samples.mean(axis=0)

            # Calculate inertial at current iteration
            inertia_active = 0
            for i in range(self.n_clusters):
                if type(X_active) == np.ndarray:
                    inertia_active += np.linalg.norm(X_active[indices == i] - new_centers_active[i], axis=1).sum()
                else:
                    inertia_active += torch.norm(X_active[indices == i] - new_centers_active[i], dim=1).sum().item()

            inertia_passive = self.messenger.recv()

            inertia = inertia_active + inertia_passive

            if self.verbose:
                print('Iteration {}, inertia: {}'.format(iter_, inertia))

            # Check if KMeans converges
            if type(X_active) == np.ndarray:
                difference_active = np.linalg.norm(new_centers_active - cur_centers, ord='fro')
            else:
                difference_active = torch.norm(new_centers_active - cur_centers, p='fro')

            difference_passive = self.messenger.recv()

            if type(X_active) == np.ndarray:
                difference = np.sqrt(np.square(difference_active) + np.square(difference_passive))
            else:
                difference = torch.sqrt(torch.square(difference_active) + torch.square(difference_passive))
            
            if difference < self.tol:
                if self.verbose:
                    print('Converged at iteration {}.\n'.format(iter_))
                break

            # ATTENSION: Avoid using direct assignment like cur_centers = new_centers_active
            # This will cause cur_centers and new_cneters to point at the same
            # object in the memory. To fix this, you must create a new object.
            cur_centers = copy.deepcopy(new_centers_active)

        return new_centers_active, indices, inertia

    def fit(self, X_active, y):
        """Using features and little labels to do clustering.
        Args:
            X_active: numpy.ndarray or torch.Tensor with shape (n_samples, n_features)
            y: List or numpy.ndarray, or torch.Tensor with shape (n_samples,).
                For index i, if y[i] equals to self.INVALID_LABEL, then X_active[i] is
                an unlabels sample.
        Returns:
            self: The estimator itself.
        """
        begin_msg = self.messenger.recv()
        self._check_params(X_active, y)

        # _, n_seed_centroids = self._init_centroids(X_active, y)

        if type(y) == np.ndarray:
            pkg = np
        elif type(y) == torch.Tensor:
            pkg = torch
        elif type(y) == list and type(X_active) == np.ndarray:
            y = np.array(y)
            pkg = np
        elif type(y) == list and type(X_active) == torch.Tensor:
            y = torch.Tensor(y)
            pkg = torch
        else:
            raise TypeError('Data type is not supported, please check it again.')

        y_unique = pkg.unique(y)
        if self.INVALID_LABEL in y_unique:
            n_seed_centroids = len(y_unique) - 1
        else:
            n_seed_centroids = len(y_unique)
        assert n_seed_centroids <= self.n_clusters, f"The number of seed centroids" \
                                                    f"should be less than the total" \
                                                    f"number of clusters."


        if n_seed_centroids == self.n_clusters:
            self.n_init = 1

        self.messenger.send(self.n_init)

        # run constrained seed KMeans n_init times in order to choose the best one
        best_inertia = None
        best_centers_active, best_indices = None, None
        for i in range(self.n_init):
            init_centers_active, _ = self._init_centroids(X_active, y)
            if self.verbose:
                print('Initialization complete')
            new_centers_active, indices, new_inertia = self._kmeans(X_active, y, init_centers_active)
            if best_inertia is None or new_inertia < best_inertia:
                best_inertia = new_inertia
                best_centers_active = new_centers_active
                best_indices = indices
                self.messenger.send(True)
            else:
                self.messenger.send(False)

        self.inertia_ = best_inertia
        self.cluster_centers_active_ = best_centers_active
        self.indices = best_indices

        return self

    def predict(self, X_active):
        """Predict the associated cluster index of samples.
        Args:
            X_active: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).
        Returns:
            indices: The associated cluster index of each sample, with shape
            (n_samples,)
        """
        n_samples = X_active.shape[0]
        indices = [-1 for _ in range(n_samples)]

        for i in range(n_samples):
            min_norm_passive = self.messenger.recv()
            if type(X_active) == np.ndarray:
                min_idx = (np.square(np.linalg.norm(self.cluster_centers_active_ - X_active[i], axis=1)) + np.square(min_norm_passive)).argmin()
            else:
                min_idx = (torch.square(torch.norm(self.cluster_centers_active_ - X_active[i], dim=1)) + torch.square(min_norm_passive)).argmin()
            indices[i] = min_idx

        if type(X_active) == np.ndarray:
            self.messenger.send(np.array(indices))
            return np.array(indices)
        else:
            self.messenger.send(torch.tensor(indices))
            return torch.tensor(indices)

    def fit_predict(self, X_active, y):
        """Convenient function."""
        return self.fit(X_active, y).predict(X_active)

    def transform(self, X_active):
        """Transform the input to the centorid space.
        Args:
            X_active: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).
        Returns:
            output: With shape (n_samples, n_clusters)
        """
        if type(X_active) == np.ndarray:
            pkg = np
        else:
            pkg = torch

        n_samples = X_active.shape[0]
        output = pkg.empty((n_samples, self.n_clusters), dtype=X_active.dtype)
        for i in range(n_samples):
            if type(X_active) == np.ndarray:
                output[i] = np.linalg.norm(self.cluster_centers_active_ - X_active[i], axis=1)
            else:
                output[i] = torch.norm(self.cluster_centers_active_ - X_active[i], dim=1)

        output_passive = self.messenger.recv()
        output += output_passive

        return output

    def fit_transform(self, X_active, y):
        """Convenient function"""
        return self.fit(X_active, y).transform(X_active)

    def score(self, X_active):
        """Opposite of the value of X_active on the K-means objective."""
        interia = 0
        n_samples = X_active.shape[0]

        for i in range(n_samples):
            if type(X_active) == np.ndarray:
                interia += np.linalg.norm(self.cluster_centers_active_ - X_active[i], axis=1).min()
            else:
                interia += torch.norm(self.cluster_centers_active_ - X_active[i], dim=1).min().item()

        interia_passive = self.messenger.recv()

        interia += interia_passive

        return -1 * interia


def plot(X_active, estimator, color_num, name):
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame()
    df['dim1'] = X_active[:, 0]
    df['dim2'] = X_active[:, 1]
    if name == 'sklearn_kmeans':
        df['y'] = estimator.labels_
    else:
        df['y'] = estimator.indices
    plt.close()
    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.1, 0.1)
    sns.scatterplot(x='dim1', y='dim2', hue=df.y.tolist(),
                    palette=sns.color_palette('hls', color_num), data=df)
    # plt.show()
    plt.savefig('./{}.png'.format(name))


if __name__ == '__main__':
    from linkefl.common.factory import messenger_factory

    active_ip = 'localhost'
    active_port = 20001
    passive_ip = 'localhost'
    passive_port = 30001

    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.ACTIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)
    print('Active party started, listening...')


    dataset_name = 'epsilon'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    _random_state = None

    active_trainset = NumpyDataset.buildin_dataset(dataset_name=dataset_name,
                                                   role=Const.ACTIVE_NAME,
                                                   root='../data',
                                                   train=True,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option,
                                                   seed=_random_state)

    print(active_trainset.features.shape)
    # print(active_trainset.features[0,:])

    X_active = active_trainset.features
    y = active_trainset.labels.tolist()
    print('y', y)
    y = [-1 for _ in range(X_active.shape[0])]
    print('y_list', y)

    # y[3], y[24] = 0, 0
    # y[11], y[19] = 1, 1
    # y[13], y[16] = 2, 2

    n_cluster = 3
    active = ActiveConstrainedSeedKMeans(messenger=_messenger, crypto_type=None, n_clusters=n_cluster, n_init=2, verbose=False)

    # begin_fit = time.time()
    # active.fit(X_active, y)
    # end_fit = time.time()

    # indices = active.indices

    # print('indices are', indices)

    # plot(X, active, name='seed_kmeans')

    begin_fit_predict = time.time()
    fit_predict = active.fit_predict(X_active, y)
    end_fit_predict = time.time()

    score = active.score(X_active)
    print('score', score)

    # print('total fit time consumed', end_fit - begin_fit)
    print('total fit_predict time consumed', end_fit_predict - begin_fit_predict)

    print('fit predict', fit_predict)

    pca_active = PCA(n_components=2) 
    pca_active.fit(X_active)
    X_active_projection = pca_active.transform(X_active)

    plot(X_active_projection, active, color_num = n_cluster, name='active_kmeans')

    print('X_active_projection', X_active_projection[0:10, :])

