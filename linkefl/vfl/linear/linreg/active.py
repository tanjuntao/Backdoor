import copy
import time

import numpy as np
from sklearn.metrics import r2_score
from termcolor import colored

from linkefl.base import BaseModelComponent
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO
from linkefl.vfl.linear import BaseLinearActive


class ActiveLinReg(BaseLinearActive, BaseModelComponent):
    def __init__(
        self,
        epochs,
        batch_size,
        learning_rate,
        messenger,
        cryptosystem,
        logger,
        *,
        penalty=Const.L2,
        reg_lambda=0.01,
        crypto_type=Const.PAILLIER,
        precision=0.001,
        random_state=None,
        using_pool=False,
        num_workers=-1,
        val_freq=1,
        saving_model=False,
        model_path="./models",
        model_name=None,
    ):
        super(ActiveLinReg, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messenger=messenger,
            cryptosystem=cryptosystem,
            logger=logger,
            penalty=penalty,
            reg_lambda=reg_lambda,
            crypto_type=crypto_type,
            precision=precision,
            random_state=random_state,
            using_pool=using_pool,
            num_workers=num_workers,
            val_freq=val_freq,
            saving_model=saving_model,
            model_path=model_path,
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

    def train(self, trainset, testset):
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instance of NumpyDataset"
        setattr(self, "x_train", trainset.features)
        setattr(self, "x_val", testset.features)
        setattr(self, "y_train", trainset.labels)
        setattr(self, "y_val", testset.labels)

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

        best_loss = float("inf")
        best_score = 0
        start_time = None
        compu_time = 0
        # Main Training Loop Here
        self.logger.log("Start collaborative model training...")
        for epoch in range(self.epochs):
            is_best = False
            all_idxes = np.arange(n_samples)
            np.random.seed(epoch)
            np.random.shuffle(all_idxes)
            batch_losses = []
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
                for msger in self.messenger:
                    passive_wx = msger.recv()
                    full_wx += passive_wx
                _begin = time.time()
                if start_time is None:
                    start_time = time.time()
                y_hat = full_wx  # no activation function
                loss = self._loss(getattr(self, "y_train")[batch_idxes], y_hat)
                residue = self._residue(getattr(self, "y_train")[batch_idxes], y_hat)

                # Active party helps passive party to calcalate gradient
                enc_residue = np.array(self.cryptosystem.encrypt_vector(residue))
                compu_time += time.time() - _begin
                for msger in self.messenger:
                    msger.send(enc_residue)
                for msger in self.messenger:
                    enc_passive_grad = msger.recv()
                    _begin = time.time()
                    passive_grad = np.array(
                        self.cryptosystem.decrypt_vector(enc_passive_grad)
                    )
                    compu_time += time.time() - _begin
                    msger.send(passive_grad)

                # Active party calculates its gradient and update model
                active_grad = self._grad(residue, batch_idxes)
                self._gradient_descent(getattr(self, "params"), active_grad)
                batch_losses.append(loss)

            # validate model performance
            if epoch % self.val_freq == 0:
                cur_loss = np.array(batch_losses).mean()
                self.logger.log(f"Epoch: {epoch}, Loss: {cur_loss}")
                result = self.validate(testset)
                val_loss, val_score = result["loss"], result["r2"]
                if val_loss < best_loss:
                    best_loss = val_loss
                    is_best = True
                if val_score > best_score:
                    best_score = val_score
                if is_best:
                    # save_params(self.params, role='bob')
                    self.logger.log("Best model updates.")
                    if self.saving_model:
                        model_params = copy.deepcopy(getattr(self, "params"))
                        NumpyModelIO.save(
                            model_params, self.model_path, self.model_name
                        )

                for msger in self.messenger:
                    msger.send(is_best)

        self.logger.log("Finish model training.")
        self.logger.log("Best validation loss: {:.5f}".format(best_loss))
        self.logger.log("Best r2_score: {:.5f}".format(best_score))
        self.logger.log("Computation time: {:.5f}".format(compu_time))
        self.logger.log("Elapsed time: {:.5f}s".format(time.time() - start_time))
        print(colored("Best validation loss: {:.5f}".format(best_loss), "red"))
        print(colored("Best r2_score: {:.5f}".format(best_score), "red"))
        print(colored("Computation time: {:.5f}".format(compu_time), "red"))
        print(colored("Elapsed time: {:.5f}s".format(time.time() - start_time), "red"))

    def validate(self, valset):
        assert isinstance(
            valset, NumpyDataset
        ), "valset should be an instance of NumpyDataset"
        active_wx = np.matmul(valset.features, getattr(self, "params"))
        full_wx = active_wx
        for msger in self.messenger:
            passive_wx = msger.recv()
            full_wx += passive_wx
        y_pred = full_wx
        loss = ((valset.labels - y_pred) ** 2).mean()
        r2 = r2_score(valset.labels, y_pred)

        return {"loss": loss, "r2": r2}

    def predict(self, testset):
        return self.validate(testset)

    def fit(self, trainset, validset, role=Const.ACTIVE_NAME):
        self.train(trainset, validset)

    def score(self, testset, role=Const.ACTIVE_NAME):
        return self.predict(testset)

    @staticmethod
    def online_inference(dataset, model_name, messenger, model_path="./models"):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        model_params = NumpyModelIO.load(model_path, model_name)
        active_wx = np.matmul(dataset.features, model_params)
        passive_wx = messenger.recv()
        y_pred = active_wx + passive_wx
        loss = ((dataset.labels - y_pred) ** 2).mean()
        r2 = r2_score(dataset.labels, y_pred)
        scores = {"loss": loss, "r2": r2}
        messenger.send([scores, y_pred])

        return scores, y_pred


if __name__ == "__main__":
    from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
    from linkefl.feature.transform import add_intercept

    # 0. Set parameters
    dataset_name = "diabetes"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = ["localhost", "localhost"]
    active_port = [20000, 30000]
    passive_ip = ["localhost", "localhost"]
    passive_port = [20001, 30001]
    _epochs = 200000
    _batch_size = -1
    _learning_rate = 1.0
    _penalty = Const.NONE
    _reg_lambda = 0.01
    _crypto_type = Const.PLAIN
    _random_state = None
    _key_size = 1024
    _val_freq = 5000

    # 1. Loading dataset and preprocessing
    # Option 1: Scikit-learn style
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root="../../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_testset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root="../../data",
        train=False,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_trainset = add_intercept(active_trainset)
    active_testset = add_intercept(active_testset)
    print("Done.")

    # Option 2: PyTorch style
    print("Loading dataset...")
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

    # 3. Initialize cryptosystem
    _crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=10000,
        gen_from_set=False,
    )

    # 4. Initialize messenger
    _messenger = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=ac_ip,
            active_port=ac_port,
            passive_ip=pass_ip,
            passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in zip(
            active_ip, active_port, passive_ip, passive_port
        )
    ]
    print("ACTIVE PARTY started, listening...")

    # 5. Initialize model and start training
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    active_party = ActiveLinReg(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messenger=_messenger,
        cryptosystem=_crypto,
        logger=_logger,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        val_freq=_val_freq,
        saving_model=False,
    )

    active_party.train(active_trainset, active_testset)

    # 6. Close messenger, finish training.
    for msger_ in _messenger:
        msger_.close()

    # # For online inference, you only need to substitue the model name
    # scores = ActiveLinReg.online_inference(
    #     active_testset,
    #     model_name='20220831_190241-active_party-vertical_linreg-402_samples.model',
    #     messenger=_messenger
    # )
    # print(scores)
