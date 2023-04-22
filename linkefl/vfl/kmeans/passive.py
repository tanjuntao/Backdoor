import copy
import datetime
import os
import pathlib
from typing import Optional

import numpy as np
import torch

from linkefl.base import BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO
from linkefl.vfl.utils.evaluate import Plot


class PassiveConstrainedSeedKMeans(BaseModelComponent):
    """Constrained seed KMeans algorithm proposed by Basu et al. in 2002."""

    def __init__(
        self,
        messenger: BaseMessenger,
        logger: GlobalLogger,
        crypto_type: str,
        n_clusters: int,
        *,
        n_init: int = 10,
        max_iter: int = 30,
        tol: float = 0.0001,
        verbose: bool = False,
        invalid_label: int = -1,
        random_state: Optional[int] = None,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
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
        self.INVALID_LABEL = invalid_label
        self.messenger = messenger
        self.logger = logger
        self.random_state = random_state
        self.saving_model = saving_model

        if self.random_state is not None:
            torch.random.manual_seed(random_state)

        if self.saving_model:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if model_dir is None:
                default_dir = "models"
                model_dir = os.path.join(default_dir, self.create_time)
            if model_name is None:
                algo_name = Const.AlgoNames.VFL_KMEANS
                model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.PASSIVE_NAME,
                        algo_name=algo_name,
                    )
                    + ".model"
                )
            self.model_dir = model_dir
            self.model_name = model_name
            self.pics_dir = self.model_dir
            if not os.path.exists(self.model_dir):
                pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def fit(self, trainset, validset, role=Const.PASSIVE_NAME):
        return self.train(trainset)

    def train(self, X_passive_dataset):
        """Using features and little labels to do clustering.
        Args:
            X_passive_dataset: NumpyDataset/TorchDataset
            X_passive: numpy.ndarray or torch.Tensor with shape (n_samples, n_features)
            y: List or numpy.ndarray, or torch.Tensor with shape (n_samples,).
                For index i, if y[i] equals to self.INVALID_LABEL, then X_passive[i] is
                an unlabels sample.
        Returns:
            self: The estimator itself.
        """
        self.messenger.send("start signal.")
        self._check_params(X_passive_dataset)

        self.n_init = self.messenger.recv()

        # run constrained seed KMeans n_init times in order to choose the best one
        best_centers_passive = None
        for i in range(self.n_init):
            init_centers_passive = self._init_centroids(X_passive_dataset)
            if self.verbose:
                print("Initialization complete")
            new_centers_passive, self.indices = self._kmeans(
                X_passive_dataset, init_centers_passive
            )
            if self.messenger.recv() is True:
                best_centers_passive = new_centers_passive

        self.cluster_centers_passive_ = best_centers_passive

        y_pred = self.score(X_passive_dataset)

        if self.saving_model:
            saved_data = (self.n_clusters, self.cluster_centers_passive_)
            NumpyModelIO.save(saved_data, self.model_dir, self.model_name)

            Plot.plot_pca(X_passive_dataset, y_pred, self.n_clusters, self.pics_dir)
            Plot.plot_silhoutte(
                X_passive_dataset, y_pred, self.n_clusters, self.pics_dir
            )

    def score(self, X_passive_dataset, role=Const.PASSIVE_NAME):
        """Predict the associated cluster index of samples.
        Args:
            X_passive_dataset: NumpyDataset/TorchDataset
            X_passive: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).
        Returns:
            indices: The associated cluster index of each sample, with shape
            (n_samples,)
        """
        X_passive = X_passive_dataset.features

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

        if type(X_passive) == np.ndarray:
            return np.array(self.indices)
        else:
            return torch.tensor(self.indices)

    @staticmethod
    def online_inference(
        dataset, messenger, logger, model_dir, model_name, role=Const.PASSIVE_NAME
    ):
        n_clusters, cluster_centers_passive_ = NumpyModelIO.load(model_dir, model_name)

        passive_model = PassiveConstrainedSeedKMeans(
            messenger=messenger,
            logger=logger,
            crypto_type="",
            n_clusters=n_clusters,
        )
        passive_model.n_clusters = n_clusters
        passive_model.cluster_centers_passive_ = cluster_centers_passive_

        y_pred = passive_model.score(dataset)

        dis = passive_model._cal_dis(dataset)
        scores = {"distance": dis}

        return scores, y_pred

    def _check_params(self, X_passive_dataset):
        """Check if the parameters of the algorithm and the inputs to it are valid."""
        X_passive = X_passive_dataset.features

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

    def _init_centroids(self, X_passive_dataset):
        """Initialize cluster centers with little samples having label."""
        X_passive = X_passive_dataset.features

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

    def _kmeans(self, X_passive_dataset, init_centers_passive):
        """KMeans algorithm implementation."""
        X_passive = X_passive_dataset.features

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

            if self.messenger.recv() == 'break':
                break

            # ATTENSION: Avoid using direct assignment
            # like cur_centers = new_centers_passive
            # This will cause cur_centers and new_cneters to point at the same
            # object in the memory. To fix this, you must create a new object.
            cur_centers = copy.deepcopy(new_centers_passive)

        return new_centers_passive, indices

    def _cal_dis(self, X_passive_dataset):
        """Opposite of the value of X_passive on the K-means objective."""
        X_passive = X_passive_dataset.features

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

    def train_predict(self, X_passive_dataset):
        """Convenient function."""
        return self.train(X_passive_dataset).predict(X_passive_dataset)

    def transform(self, X_passive_dataset):
        """Transform the input to the centorid space.
        Args:
            X_passive_dataset: Numpydataset/TorchDataset
            X_passive: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).
        Returns:
            output_passive: With shape (n_samples, n_clusters)
        """
        X_passive = X_passive_dataset.features

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

    def train_transform(self, X_passive_dataset):
        """Convenient function"""
        return self.train(X_passive_dataset).transform(X_passive_dataset)

    def _save_model(self):
        if self.saving_model:
            saved_data = [self.n_clusters, self.cluster_centers_passive_]
            NumpyModelIO.save(saved_data, self.model_dir, self.model_name)

    def load_model(self, model_path="./models", model_name="vfl_kmeans_passive"):
        self.n_clusters, self.cluster_centers_passive_ = NumpyModelIO.load(
            model_path, model_name
        )
        return self.n_clusters, self.cluster_centers_passive_


if __name__ == "__main__":
    from linkefl.common.factory import logger_factory, messenger_factory

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
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    dataset_name = "epsilon"
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

    n_cluster = 2
    passive = PassiveConstrainedSeedKMeans(
        messenger=_messenger,
        logger=_logger,
        crypto_type="",
        n_clusters=n_cluster,
        n_init=2,
        verbose=False,
        saving_model=True,
    )

    passive.fit(passive_trainset, passive_trainset)

    # scores, y_pred = PassiveConstrainedSeedKMeans.online_inference(
    #     passive_trainset, _messenger, _logger,
    #     "./models/20230321170106", "20230321170106-passive_party-vfl_kmeans.model",
    #     Const.PASSIVE_NAME
    # )
    # print(scores, y_pred)
