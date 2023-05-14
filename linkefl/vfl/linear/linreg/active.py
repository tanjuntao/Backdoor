import copy
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from termcolor import colored

from linkefl.base import BaseCryptoSystem, BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO
from linkefl.vfl.linear.base import BaseLinearActive
from linkefl.vfl.tree.plotting import Plot


class ActiveLinReg(BaseLinearActive, BaseModelComponent):
    def __init__(
        self,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        messengers: List[BaseMessenger],
        cryptosystem: BaseCryptoSystem,
        logger: GlobalLogger,
        rank: int = 0,
        penalty: str = "l2",
        reg_lambda: float = 0.01,
        num_workers: int = 1,
        val_freq: int = 1,
        random_state: Optional[int] = None,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        super(ActiveLinReg, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messengers=messengers,
            cryptosystem=cryptosystem,
            logger=logger,
            rank=rank,
            penalty=penalty,
            reg_lambda=reg_lambda,
            num_workers=num_workers,
            val_freq=val_freq,
            random_state=random_state,
            saving_model=saving_model,
            model_dir=model_dir,
            model_name=model_name,
            task="regression",
        )

    def _loss(self, y_true, y_hat):
        # Linear regression uses MSE-loss as loss function
        train_loss = ((y_true - y_hat) ** 2).mean()

        params = getattr(self, "params")
        if self.penalty == Const.NONE:
            reg_loss = 0.0
        elif self.penalty == Const.L1:
            reg_loss = self.reg_lambda * abs(params).sum()
        elif self.penalty == Const.L2:
            reg_loss = 1.0 / 2 * self.reg_lambda * (params**2).sum()
        else:
            raise ValueError("Regularization method not supported now.")
        total_loss = train_loss + reg_loss

        return total_loss

    def train(self, trainset: NumpyDataset, validset: NumpyDataset) -> None:
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            validset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"
        setattr(self, "x_train", trainset.features)
        setattr(self, "x_val", validset.features)
        setattr(self, "y_train", trainset.labels)
        setattr(self, "y_val", validset.labels)

        # initialize model parameters
        params = self._init_weights(trainset.n_features)
        setattr(self, "params", params)

        # trainfer public key to passive party
        self._sync_pubkey()

        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        best_loss, best_score = float("inf"), 0.0
        residue_records = []
        r2_records = []
        train_loss_records, valid_loss_records = [], []
        mae_records, mse_records, sse_records = [], [], []
        start_time = None
        # Main Training Loop Here
        self.logger.log("Start collaborative model training...")
        for epoch in range(self.epochs):
            all_idxes = np.arange(n_samples)
            np.random.seed(epoch)
            np.random.shuffle(all_idxes)
            batch_losses, batch_residues = [], []
            for batch in range(n_batches):
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Active party calculates loss and residue
                active_wx = np.matmul(
                    getattr(self, "x_train")[batch_idxes], getattr(self, "params")
                )
                full_wx = active_wx
                for msger in self.messengers:
                    passive_wx = msger.recv()
                    full_wx += passive_wx
                if start_time is None:
                    start_time = time.time()
                y_hat = full_wx  # no activation function
                loss = self._loss(getattr(self, "y_train")[batch_idxes], y_hat)
                residue = self._residue(getattr(self, "y_train")[batch_idxes], y_hat)

                # Active party helps passive party to calcalate gradient
                enc_residue = np.array(self.cryptosystem.encrypt_vector(residue))
                for msger in self.messengers:
                    msger.send(enc_residue)
                for msger in self.messengers:
                    enc_passive_grad = msger.recv()
                    passive_grad = np.array(
                        self.cryptosystem.decrypt_vector(enc_passive_grad)
                    )
                    msger.send(passive_grad)

                # Active party calculates its gradient and update model
                active_grad = self._grad(residue, batch_idxes)
                self._gradient_descent(getattr(self, "params"), active_grad)
                batch_losses.append(loss)
                batch_residues.append(residue.mean())

            # validate model performance
            is_best = False
            if (epoch + 1) % self.val_freq == 0:
                scores = self.validate(validset)
                train_scores = self.validate(  # noqa: F841,
                    trainset
                )  # do not delete this line
                train_loss = np.array(batch_losses).mean()
                self.logger.log(
                    f"Epoch: {epoch}, Train loss: {train_loss}, Valid loss:"
                    f" {scores['loss']}"
                )

                residue_records.append(np.array(batch_residues).mean())
                train_loss_records.append(train_loss)
                valid_loss_records.append(scores["loss"])
                mae_records.append(scores["mae"])
                mse_records.append(scores["mse"])
                sse_records.append(scores["sse"])
                r2_records.append(scores["r2"])
                if scores["loss"] < best_loss:
                    best_loss = scores["loss"]
                    is_best = True
                if scores["r2"] > best_score:
                    best_score = scores["r2"]
                    is_best = True
                self.logger.log_metric(
                    epoch=epoch + 1,
                    loss=scores["loss"],
                    mae=scores["mae"],
                    mse=scores["mse"],
                    sse=scores["sse"],
                    r2=scores["r2"],
                    total_epoch=self.epochs,
                )
                if is_best:
                    print(colored("Best model updates.", "red"))
                    self.logger.log("Best model updates.")
                    if self.saving_model:
                        model_params = copy.deepcopy(getattr(self, "params"))
                        NumpyModelIO.save(model_params, self.model_dir, self.model_name)
                for msger in self.messengers:
                    msger.send(is_best)

        if self.executor_pool is not None:
            self.executor_pool.close()
            self.executor_pool.join()

        if self.saving_model:
            Plot.plot_residual(residue_records, self.pics_dir)
            Plot.plot_train_test_loss(
                train_loss_records, valid_loss_records, self.pics_dir
            )
            Plot.plot_regression_metrics(
                mae_records, mse_records, sse_records, r2_records, self.pics_dir
            )

        self.logger.log("Finish model training.")
        self.logger.log("Best validation loss: {:.5f}".format(best_loss))
        self.logger.log("Best r2_score: {:.5f}".format(best_score))
        self.logger.log("Elapsed time: {:.5f}s".format(time.time() - start_time))
        print(colored("Best validation loss: {:.5f}".format(best_loss), "red"))
        print(colored("Best r2_score: {:.5f}".format(best_score), "red"))
        print(colored("Elapsed time: {:.5f}s".format(time.time() - start_time), "red"))

    def validate(self, validset: NumpyDataset) -> Dict[str, float]:
        assert isinstance(
            validset, NumpyDataset
        ), "valset should be an instance of NumpyDataset"
        active_wx = np.matmul(validset.features, getattr(self, "params"))
        full_wx = active_wx
        for msger in self.messengers:
            passive_wx = msger.recv()
            full_wx += passive_wx
        y_pred = full_wx
        loss = ((validset.labels - y_pred) ** 2).mean()
        mae = mean_absolute_error(validset.labels, y_pred)
        mse = mean_squared_error(validset.labels, y_pred)
        sse = mse * len(validset.labels)
        r2 = r2_score(validset.labels, y_pred)
        scores = {"loss": loss, "mae": mae, "mse": mse, "sse": sse, "r2": r2}

        return scores

    def predict(self, testset: NumpyDataset) -> Dict[str, float]:
        return self.validate(testset)

    def fit(
        self,
        trainset: NumpyDataset,
        validset: NumpyDataset,
        role: str = Const.ACTIVE_NAME,
    ) -> None:
        self.train(trainset, validset)

    def score(
        self, testset: NumpyDataset, role: str = Const.ACTIVE_NAME
    ) -> Dict[str, float]:
        return self.predict(testset)

    @staticmethod
    def online_inference(
        dataset: NumpyDataset,
        messengers: List[BaseMessenger],
        logger: GlobalLogger,
        model_dir: str,
        model_name: str,
        role: str = Const.ACTIVE_NAME,
    ):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        model_params = NumpyModelIO.load(model_dir, model_name)
        active_wx = np.matmul(dataset.features, model_params)
        total_wx = active_wx
        for messenger in messengers:
            curr_wx = messenger.recv()
            total_wx += curr_wx
        y_pred = total_wx

        return y_pred


if __name__ == "__main__":
    from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
    from linkefl.feature.transform import add_intercept

    # Set parameters
    _dataset_name = "diabetes"
    _passive_feat_frac = 0.5
    _feat_perm_option = Const.SEQUENCE
    _active_ips = [
        "localhost",
    ]
    _active_ports = [
        20000,
    ]
    _passive_ips = [
        "localhost",
    ]
    _passive_ports = [
        30000,
    ]
    _epochs = 200000
    _batch_size = -1
    _learning_rate = 1.0
    _penalty = Const.NONE
    _reg_lambda = 0.01
    _crypto_type = Const.PLAIN
    _random_state = None
    _key_size = 1024
    _val_freq = 5000
    _num_workers = 1
    _saving_model = True
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=10,
        gen_from_set=False,
    )
    _messengers = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=ac_ip,
            active_port=ac_port,
            passive_ip=pass_ip,
            passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in zip(
            _active_ips, _active_ports, _passive_ips, _passive_ports
        )
    ]

    # Loading dataset and preprocessing
    # Option 1: Scikit-learn style
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=True,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
    )
    active_testset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=False,
        download=True,
        passive_feat_frac=_passive_feat_frac,
        feat_perm_option=_feat_perm_option,
    )
    active_trainset = add_intercept(active_trainset)
    active_testset = add_intercept(active_testset)
    print("Done.")

    # Option 2: PyTorch style
    # print("Loading dataset...")
    # transform = AddIntercept()
    # active_trainset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
    #                                                dataset_name=dataset_name,
    #                                                train=True,
    #                                                passive_feat_frac=passive_feat_frac,
    #                                                feat_perm_option=feat_perm_option,
    #                                                transform=transform)
    # active_testset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
    #                                               dataset_name=dataset_name,
    #                                               train=False,
    #                                               passive_feat_frac=passive_feat_frac,
    #                                               feat_perm_option=feat_perm_option,
    #                                               transform=transform)
    # print('Done.')

    # Initialize model and start training
    print("ACTIVE PARTY started, listening...")
    active_party = ActiveLinReg(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messengers=_messengers,
        cryptosystem=_crypto,
        logger=_logger,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        num_workers=_num_workers,
        random_state=_random_state,
        val_freq=_val_freq,
        saving_model=_saving_model,
    )
    active_party.train(active_trainset, active_testset)

    # Close messengers, finish training.
    for msger_ in _messengers:
        msger_.close()
