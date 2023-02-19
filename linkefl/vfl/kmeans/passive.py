import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset


class PassiveConstrainedSeedKMeans:
    """Constrained seed KMeans algorithm proposed by Basu et al. in 2002."""

    def __init__(
        self,
        messenger,
        crypto_type,
        n_clusters=2,
        *,
        n_init=10,
        max_iter=30,
        tol=0.0001,
        verbose=False,
        invalide_label=-1,
        random_state=0,
    ):
        """Initialization a constrained seed kmeans estimator.
        Args:
            n_clusters: The number of clusters.
            n_init: The number of times the algorithm will run in order to choose
                the best result.
            max_iter: The maximum number of iterations the algorithm will run.
            tol: The convergence threshold of the algorithm. If the norm of a
                matrix, which is the difference_passive between two consective cluster
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

    def _check_params(self, X_passive):
        """Check if the parameters of the algorithm and the inputs to it are valid."""
        if type(X_passive) not in (np.ndarray, torch.Tensor):
            raise TypeError(
                "Type of X_passive can only take numpy.ndarray and "
                f"torch.Tensor, but got {type(X_passive)} instead."
            )

        if self.n_clusters > X_passive.shape[0]:
            raise ValueError(
                "The number of clusters mube be less than the number of samples."
            )

        if self.max_iter <= 0:
            raise ValueError("The number of maximum iteration must larger than zero.")

    def _init_centroids(self, X_passive):
        """Initialize cluster centers with little samples having label."""

        if type(X_passive) == np.ndarray:
            pkg = np
        elif type(X_passive) == torch.Tensor:
            pkg = torch
        else:
            raise TypeError("Data type is not supported, please check it again.")

        n_seed_centroids = self.messenger.recv()

        if self.verbose:
            print("file receive", n_seed_centroids)

        labeled_idxes = self.messenger.recv()

        centers_passive = pkg.empty(
            (self.n_clusters, X_passive.shape[1]), dtype=X_passive.dtype
        )
        # First, initialize seed centers using samples with label
        for i in range(n_seed_centroids):
            seed_samples = X_passive[labeled_idxes[i]]
            centers_passive[i] = seed_samples.mean(axis=0)

        if self.verbose:
            print("centers_passive", centers_passive)

        # # Then, initilize the remaining centers with random samples from X_passive
        unlabel_idxes = self.messenger.recv()  # np.where returns a tuple
        if self.verbose:
            print("unlabeled index", unlabel_idxes)

        if len(unlabel_idxes) < self.n_clusters - n_seed_centroids:
            np.random.seed(self.random_state)
            idx = np.random.randint(
                X_passive.shape[0], size=self.n_clusters - n_seed_centroids
            )

            for i in range(n_seed_centroids, self.n_clusters):
                centers_passive[i] = X_passive[idx[i - n_seed_centroids]]
        else:
            for i in range(n_seed_centroids, self.n_clusters):
                np.random.seed(self.random_state)
                idx = np.random.choice(unlabel_idxes, 1, replace=False)
                centers_passive[i] = X_passive[idx]

        if self.verbose:
            print("final passive centers:", centers_passive)

        return centers_passive

    def _kmeans(self, X_passive, init_centers_passive):
        """KMeans algorithm implementation."""
        # indices = copy.copy(y)
        # if type(indices) == list:
        #     indices = np.array(indices)
        n_samples = X_passive.shape[0]
        cur_centers = init_centers_passive
        new_centers_passive = copy.deepcopy(init_centers_passive)

        valid_idxes = self.messenger.recv()

        # Main loop
        for iter_ in range(self.max_iter):
            # Fist step in KMeans: calculate the closest centroid for each sample
            for i in range(n_samples):
                # If this sample has label, then we use the ground-truth label
                # as its cluster index
                if i in valid_idxes:
                    continue

                # print(i)

                # if type(X_passive) == np.ndarray:
                #     min_idx = np.linalg.norm(
                #         cur_centers - X_passive[i],
                #         axis=1
                #     ).argmin()
                # else:
                #     min_idx = torch.norm(cur_centers - X_passive[i], dim=1).argmin()
                # indices[i] = min_idx

                if type(X_passive) == np.ndarray:
                    passive_norm = np.linalg.norm(cur_centers - X_passive[i], axis=1)
                else:
                    passive_norm = torch.norm(cur_centers - X_passive[i], dim=1)

                # print('min norm', passive_norm)
                self.messenger.send(passive_norm)

            indices = self.messenger.recv()

            # Second step in KMeans: update each centroids
            for i in range(self.n_clusters):
                passive_cluster_samples = X_passive[indices == i]
                # In the case that the cluster is empty, randomly choose
                # a sample from X_passive.
                if passive_cluster_samples.shape[0] == 0:
                    np.random.seed(self.random_state)
                    new_centers_passive[i] = X_passive[
                        np.random.choice(n_samples, 1, replace=False)
                    ]
                else:
                    new_centers_passive[i] = passive_cluster_samples.mean(axis=0)

            # Calculate inertial at current iteration
            inertia_passive = 0
            for i in range(self.n_clusters):
                if type(X_passive) == np.ndarray:
                    inertia_passive += np.linalg.norm(
                        X_passive[indices == i] - new_centers_passive[i], axis=1
                    ).sum()
                else:
                    inertia_passive += (
                        torch.norm(
                            X_passive[indices == i] - new_centers_passive[i], dim=1
                        )
                        .sum()
                        .item()
                    )

            self.messenger.send(inertia_passive)

            # if self.verbose:
            #     print('Iteration {}, inertia_passive: {}'
            #           .format(iter_, inertia_passive))

            # Check if KMeans converges
            if type(X_passive) == np.ndarray:
                difference_passive = np.linalg.norm(
                    new_centers_passive - cur_centers, ord="fro"
                )
            else:
                difference_passive = torch.norm(
                    new_centers_passive - cur_centers, p="fro"
                )

            self.messenger.send(difference_passive)

            if difference_passive < self.tol:
                if self.verbose:
                    print("Converged at iteration {}.\n".format(iter_))
                break

            # ATTENSION: Avoid using direct assignment
            # like cur_centers = new_centers_passive
            # This will cause cur_centers and new_cneters to point at the same
            # object in the memory. To fix this, you must create a new object.
            cur_centers = copy.deepcopy(new_centers_passive)

        return new_centers_passive, indices

    def fit(self, X_passive):
        """Using features and little labels to do clustering.
        Args:
            X_passive: numpy.ndarray or torch.Tensor with shape (n_samples, n_features)
            y: List or numpy.ndarray, or torch.Tensor with shape (n_samples,).
                For index i, if y[i] equals to self.INVALID_LABEL, then X_passive[i] is
                an unlabels sample.
        Returns:
            self: The estimator itself.
        """
        self.messenger.send("start signal.")
        self._check_params(X_passive)

        self.n_init = self.messenger.recv()

        # run constrained seed KMeans n_init times in order to choose the best one
        best_centers_passive = None
        for i in range(self.n_init):
            init_centers_passive = self._init_centroids(X_passive)
            if self.verbose:
                print("Initialization complete")
            new_centers_passive, indices = self._kmeans(X_passive, init_centers_passive)
            if self.messenger.recv() is True:
                best_centers_passive = new_centers_passive

        self.cluster_centers_passive_ = best_centers_passive

        return self

    def predict(self, X_passive):
        """Predict the associated cluster index of samples.
        Args:
            X_passive: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).
        Returns:
            indices: The associated cluster index of each sample, with shape
            (n_samples,)
        """
        n_samples = X_passive.shape[0]
        # indices = [-1 for _ in range(n_samples)]

        for i in range(n_samples):
            if type(X_passive) == np.ndarray:
                min_norm_passive = np.linalg.norm(
                    self.cluster_centers_passive_ - X_passive[i], axis=1
                )
            else:
                min_norm_passive = torch.norm(
                    self.cluster_centers_passive_ - X_passive[i], dim=1
                )
            self.messenger.send(min_norm_passive)

        self.indices = self.messenger.recv()
        #     indices[i] = min_idx

        # if type(X_passive) == np.ndarray:
        #     return np.array(indices)
        # else:
        #     return torch.tensor(indices)

    def fit_predict(self, X_passive):
        """Convenient function."""
        return self.fit(X_passive).predict(X_passive)

    def transform(self, X_passive):
        """Transform the input to the centorid space.
        Args:
            X_passive: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).
        Returns:
            output_passive: With shape (n_samples, n_clusters)
        """
        if type(X_passive) == np.ndarray:
            pkg = np
        else:
            pkg = torch

        n_samples = X_passive.shape[0]
        output_passive = pkg.empty((n_samples, self.n_clusters), dtype=X_passive.dtype)
        for i in range(n_samples):
            if type(X_passive) == np.ndarray:
                output_passive[i] = np.linalg.norm(
                    self.cluster_centers_passive_ - X_passive[i], axis=1
                )
            else:
                output_passive[i] = torch.norm(
                    self.cluster_centers_passive_ - X_passive[i], dim=1
                )

        self.messenger.send(output_passive)

        return output_passive

    def fit_transform(self, X_passive):
        """Convenient function"""
        return self.fit(X_passive).transform(X_passive)

    def score(self, X_passive):
        """Opposite of the value of X_passive on the K-means objective."""
        interia_passive = 0
        n_samples = X_passive.shape[0]

        for i in range(n_samples):
            if type(X_passive) == np.ndarray:
                interia_passive += np.linalg.norm(
                    self.cluster_centers_passive_ - X_passive[i], axis=1
                ).min()
            else:
                interia_passive += (
                    torch.norm(self.cluster_centers_passive_ - X_passive[i], dim=1)
                    .min()
                    .item()
                )

        self.messenger.send(interia_passive)

        return -1 * interia_passive


def pca_plot(X_passive, estimator, x_lim_left, x_lim_right, y_lim_down, y_lim_up, color_num, name):
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame()
    df["dim1"] = X_passive[:, 0]
    df["dim2"] = X_passive[:, 1]
    if name == "sklearn_kmeans":
        df["y"] = estimator.labels_
    else:
        df["y"] = estimator.indices
    plt.close()
    plt.xlim(x_lim_left, x_lim_right)
    plt.ylim(y_lim_down, y_lim_up)
    sns.scatterplot(
        x="dim1",
        y="dim2",
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", color_num),
        data=df,
    )
    # plt.show()
    plt.savefig("./{}.png".format(name))
    plt.close()


def sil_plot(X_passive, estimator, n_cluster, name):
    silhouette_values = silhouette_samples(X_passive, estimator.indices)
    sil_per_cls = [[], [], []]
    for cls_idx in range(n_cluster):
        for idx in range(len(estimator.indices)):
            if estimator.indices[idx] == cls_idx:
                sil_per_cls[cls_idx].append(silhouette_values[idx])
    
    plt.boxplot(sil_per_cls)
    # plt.show()
    plt.title('Silhouette Coefficient Distribution for Each Cluster')
    plt.savefig("./{}_silhoutte.png".format(name))
    plt.close()



if __name__ == "__main__":
    from linkefl.common.factory import messenger_factory

    active_ip = "localhost"
    active_port = 20001
    passive_ip = "localhost"
    passive_port = 30001

    _messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    dataset_name = "digits"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    _random_state = None

    passive_trainset = NumpyDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.PASSIVE_NAME,
        root="../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
        seed=_random_state,
    )

    print(passive_trainset.features.shape)

    X_passive = passive_trainset.features

    n_cluster = 3
    passive = PassiveConstrainedSeedKMeans(
        messenger=_messenger,
        crypto_type=None,
        n_clusters=n_cluster,
        n_init=2,
        verbose=False,
    )

    passive.fit_predict(X_passive)

    passive.score(X_passive)

    pca_passive = PCA(n_components=2)
    pca_passive.fit(X_passive)
    X_passive_projection = pca_passive.transform(X_passive)

    x_lim_left = 1.2 * X_passive_projection[:, 0].min()
    x_lim_right = 1.2 * X_passive_projection[:, 0].max()
    y_lim_down = 1.2 * X_passive_projection[:, 1].min()
    y_lim_up = 1.2 * X_passive_projection[:, 1].max()

    pca_plot(X_passive_projection, passive, 
             x_lim_left=x_lim_left,
             x_lim_right=x_lim_right,
             y_lim_down=y_lim_down,
             y_lim_up=y_lim_up,
             color_num=n_cluster, name="{}_passive_kmeans".format(dataset_name))
    
    sil_plot(X_passive=X_passive, estimator=passive, n_cluster=n_cluster, name='{}_passive_kmeans'.format(dataset_name))

    # print("X_passive_projection", X_passive_projection[0:10, :])
