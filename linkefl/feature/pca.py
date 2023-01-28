import numpy as np
import torch
from sklearn.decomposition import PCA as SklearnPCA

from linkefl.base import BaseTransformComponent
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset


class PCA(BaseTransformComponent):
    """
    Ref:https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

        .. versionadded:: 0.18.0

    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

        .. versionadded:: 0.18.0

    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).

        .. versionadded:: 0.18.0

    n_oversamples : int, default=10
        This parameter is only relevant when `svd_solver="randomized"`.
        It corresponds to the additional number of random vectors to sample the
        range of `X` so as to ensure proper conditioning. See
        :func:`~sklearn.utils.extmath.randomized_svd` for more details.

        .. versionadded:: 1.1

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Power iteration normalizer for randomized SVD solver.
        Not used by ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd`
        for more details.

        .. versionadded:: 1.1

    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

        .. versionadded:: 0.18.0

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. Equivalently, the right singular
        vectors of the centered input data, parallel to its eigenvectors.
        The components are sorted by ``explained_variance_``.

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
        The variance estimation uses `n_samples - 1` degrees of freedom.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

        .. versionadded:: 0.18

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

        .. versionadded:: 0.19

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    n_features_ : int
        Number of features in the training data.

    n_samples_ : int
        Number of samples in the training data.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        compute the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    """

    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    def __call__(self, dataset, role):
        pca = SklearnPCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )

        offset = 2 if dataset.role == Const.ACTIVE_NAME else 1
        raw_dataset = dataset.get_dataset()
        if isinstance(raw_dataset, np.ndarray):
            X_transform = pca.fit_transform(raw_dataset[:, offset:])
            raw_dataset = np.concatenate((raw_dataset[:, :offset], X_transform), axis=1)
            # raw_dataset[:, offset:] = X_transform
        elif isinstance(raw_dataset, torch.Tensor):
            X_transform = pca.fit_transform(raw_dataset[:, offset:].numpy())
            raw_dataset = torch.cat(
                (raw_dataset[:, :offset], torch.from_numpy(X_transform)), dim=1
            )
            # raw_dataset[:, offset:] = torch.from_numpy(X_transform)
        else:
            raise TypeError("invalid datatype")

        dataset.set_dataset(raw_dataset)
        self.raw_pca = pca

        return dataset

    @property
    def components_(self):
        return self.raw_pca.components_

    @property
    def explained_variance_(self):
        return self.raw_pca.explained_variance_

    @property
    def explained_variance_ratio_(self):
        return self.raw_pca.explained_variance_ratio_

    @property
    def singular_values_(self):
        return self.raw_pca.singular_values_

    @property
    def mean_(self):
        return self.raw_pca.mean_

    @property
    def n_components_(self):
        return self.raw_pca.n_components_

    @property
    def n_features_(self):
        return self.raw_pca.n_features_

    @property
    def n_samples_(self):
        return self.raw_pca.n_samples_

    @property
    def noise_variance_(self):
        return self.raw_pca.noise_variance_

    @property
    def n_features_in_(self):
        return self.raw_pca.n_features_in_


if __name__ == "__main__":
    dataset_name = "credit"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    _random_state = None

    active_trainset = NumpyDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.ACTIVE_NAME,
        root="data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
        seed=_random_state,
    )

    print(f"before: {active_trainset.features.shape}")
    instance = PCA(n_components=2)
    _X_transform = instance(active_trainset, role=Const.ACTIVE_NAME)
    print(f"after: {active_trainset.features.shape}")

    print("X_transform", type(_X_transform))
    print("instance_pca", instance.explained_variance_ratio_)

    print("components", instance.components_)
    print("explained_variance_", instance.explained_variance_)
    print("explained_variance_ratio_", instance.explained_variance_ratio_)
    print("sigular values", instance.singular_values_)
    print("mean_", instance.mean_)
    print("n_components_", instance.n_components_)
    print("n_features_", instance.n_features_)
    print("n_samples_", instance.n_samples_)
    print("noise_variance_", instance.noise_variance_)
    print("n_features_in_", instance.n_features_in_)
