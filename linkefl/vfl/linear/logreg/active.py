import copy
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from termcolor import colored

from linkefl.base import BaseModelComponent
from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO
from linkefl.util import sigmoid
from linkefl.vfl.linear import BaseLinearActive
from linkefl.vfl.tree.plotting import Plot
from linkefl.vfl.tree.loss_functions import CrossEntropyLoss

class ActiveLogReg(BaseLinearActive, BaseModelComponent):
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
        positive_thresh=0.5,
        residue_precision=0.0001,
    ):
        super(ActiveLogReg, self).__init__(
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
            task="classification",
        )
        self.POSITIVE_THRESH = positive_thresh
        self.RESIDUE_PRECISION = len(str(residue_precision).split(".")[1])

    @staticmethod
    def _logloss(y_true, y_hat):
        origin_size = len(y_true)
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                y_true = np.append(y_true, 1)
                y_hat = np.append(y_hat, 1.0)
            else:
                y_true = np.append(y_true, 0)
                y_hat = np.append(y_hat, 0.0)

        return log_loss(y_true=y_true, y_pred=y_hat, normalize=False) / origin_size

    def _loss(self, y_true, y_hat):
        # Logistic regression uses log-loss as loss function
        train_loss = ActiveLogReg._logloss(y_true, y_hat)

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

    # @profile
    def train(self, trainset, testset):
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instanceof NumpyDataset"
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

        best_acc, best_auc = 0.0, 0.0
        start_time = None
        compu_time = 0
        # Main Training Loop Here
        self.logger.log("Start collaborative model training...")
        residual_record, train_loss_record, test_loss_record = [], [], []
        train_auc_record, test_auc_record, train_acc_record, test_acc_record, f1_record = [], [], [], [], []

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.logger.log("Epoch: {}".format(epoch))
            is_best = False
            all_idxes = np.arange(n_samples)
            np.random.seed(epoch)
            np.random.shuffle(all_idxes)
            batch_losses, batch_residuales = [], []
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
                y_hat = sigmoid(full_wx)  # use sigmoid as activation function
                loss = self._loss(getattr(self, "y_train")[batch_idxes], y_hat)
                residue = self._residue(getattr(self, "y_train")[batch_idxes], y_hat)
                # NB: In verticalLR model, the residue (equals y_true - y_hat) may be
                # very close to zero, e.g., r = 0.000000000000...000001(50 dicimal bits)
                # then the exponent term of the encrypted residue
                # will be extreamly small,
                # e.g., -50, which will cause slow the ciphertext addition operation.
                # So you should round the residue's precision before encryption.
                if self.using_pool:
                    residue = np.array(
                        [round(res, self.RESIDUE_PRECISION) for res in residue]
                    )

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
                batch_residuales.append(residue)

            # validate model performance
            if epoch % self.val_freq == 0:
                cur_loss = np.array(batch_losses).mean()
                cur_residue = np.array(batch_residuales).mean()
                self.logger.log(f"Epoch: {epoch}, Loss: {cur_loss}")

                scores = self.validate(testset, epoch)
                train_scores = self.validate(trainset, epoch)

                residual_record.append(cur_residue)
                f1_record.append(scores["f1"])
                train_loss_record.append(train_scores["loss"])
                test_loss_record.append(scores["loss"])
                train_auc_record.append(train_scores["auc"])
                test_auc_record.append(scores["auc"])
                train_acc_record.append(train_scores["acc"])
                test_acc_record.append(scores["acc"])
                print("2")
                if scores["acc"] > best_acc:
                    best_acc = scores["acc"]
                    is_best = True
                if scores["auc"] > best_auc:
                    best_auc = scores["auc"]
                    is_best = True
                self.logger.log_metric(
                    epoch,
                    cur_loss,
                    scores["acc"],
                    scores["auc"],
                    scores["f1"],
                    total_epoch=self.epochs,
                )
                print("3")
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
            print(
                colored("epoch time: {}".format(time.time() - epoch_start_time), "red")
            )

        # close ThreadPool if it exists
        if self.executor_pool is not None:
            self.executor_pool.close()
            self.executor_pool.join()

        self.logger.log("Finish model training.")
        self.logger.log("Best history acc: {:.5f}".format(best_acc))
        self.logger.log("Best history auc: {:.5f}".format(best_auc))
        self.logger.log("Computation time: {:.5f}".format(compu_time))
        self.logger.log("Elapsed time: {:.5f}s".format(time.time() - start_time))
        print(colored("Best history acc: {:.5f}".format(best_acc), "red"))
        print(colored("Best history auc: {:.5f}".format(best_auc), "red"))
        print(colored("Computation time: {:.5f}".format(compu_time), "red"))
        print(colored("Elapsed time: {:.5f}s".format(time.time() - start_time), "red"))

        scores = self.validate(testset)
        Plot.plot_residual(residual_record, self.model_path)
        Plot.plot_train_test_loss(train_loss_record, test_loss_record, self.model_path)
        Plot.plot_ordered_lorenz_curve(label=testset.labels, y_prob=scores["probs"], file_dir=self.model_path)
        Plot.plot_predict_distribution(y_prob=scores["probs"], bins=10, file_dir=self.model_path)
        Plot.plot_predict_prob_box(y_prob=scores["probs"], file_dir=self.model_path)
        Plot.plot_train_test_auc(train_auc_record, test_auc_record, self.model_path)
        Plot.plot_binary_mertics(testset.labels, scores["probs"], self.model_path)
        Plot.plot_f1_score(f1_record)

    def validate(self, valset, epoch=-1):
        assert isinstance(
            valset, NumpyDataset
        ), "valset should be an instance of NumpyDataset"
        active_wx = np.matmul(valset.features, getattr(self, "params"))
        full_wx = active_wx
        for msger in self.messenger:
            passive_wx = msger.recv()
            full_wx += passive_wx
        probs = sigmoid(full_wx)
        preds = (probs > self.POSITIVE_THRESH).astype(np.int32)

        loss = np.array(self._loss(valset.labels, preds)).mean()
        accuracy = accuracy_score(valset.labels, preds)
        f1 = f1_score(valset.labels, preds)
        auc = roc_auc_score(valset.labels, probs)

        if epoch == self.epochs - 1:
            from linkefl.vfl.tree.plotting import Plot

            Plot.plot_binary_mertics(valset.labels, probs, self.pics_path)

        return {"loss": loss, "probs": probs, "preds": preds, "acc": accuracy, "f1": f1, "auc": auc}

    def predict(self, testset):
        return self.validate(testset)

    def fit(self, trainset, validset, role=Const.ACTIVE_NAME):
        self.train(trainset, validset)

    def score(self, testset, role=Const.ACTIVE_NAME):
        return self.predict(testset)

    @staticmethod
    def online_inference(
        dataset, model_name, messenger, model_path="./models", POSITIVE_THRESH=0.5
    ):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        model_params = NumpyModelIO.load(model_path, model_name)
        active_wx = np.matmul(dataset.features, model_params)
        total_wx = active_wx
        for msger in messenger:
            curr_wx = msger.recv()
            total_wx += curr_wx
        probs = sigmoid(total_wx)
        preds = (probs > POSITIVE_THRESH).astype(np.int32)
        accuracy = accuracy_score(dataset.labels, preds)
        f1 = f1_score(dataset.labels, preds)
        auc = roc_auc_score(dataset.labels, probs)

        scores = {"acc": accuracy, "auc": auc, "f1": f1}
        for msger in messenger:
            msger.send([scores, preds])
        return scores, preds


if __name__ == "__main__":
    from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
    from linkefl.feature.transform import add_intercept, parse_label, scale

    # 0. Set parameters
    _dataset_name = "epsilon"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = [
        "localhost",
    ]
    active_port = [
        20000,
    ]
    passive_ip = [
        "localhost",
    ]
    passive_port = [
        20001,
    ]
    _epochs = 10
    _batch_size = -1
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.01
    _crypto_type = Const.PLAIN
    _random_state = 3347
    _key_size = 1024
    _using_pool = False

    # 1. Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_testset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=False,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    # load dummy dataset
    # dummy_dataset = NumpyDataset.dummy_daaset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_type=Const.CLASSIFICATION,
    #     n_samples=100000,
    #     n_features=100,
    #     passive_feat_frac=passive_feat_frac
    # )
    # active_trainset, active_testset = NumpyDataset.train_test_split(
    #     dummy_dataset,
    #     test_size=0.2
    # )

    # if using credit dataset, remember to apply scale after add_intercept,
    # otherwise the model cannot converge
    active_trainset = add_intercept(scale(parse_label(active_trainset)))
    active_testset = add_intercept(scale(parse_label(active_testset)))
    print("Done.")
    # Option 2: PyTorch style
    # print('Loading dataset...')
    # transform = Compose([ParseLabel(), Scale(), AddIntercept()])
    # active_trainset = NumpyDataset.buildin_dataset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_name=dataset_name,
    #     train=True,
    #     passive_feat_frac=passive_feat_frac,
    #     feat_perm_option=feat_perm_option,
    #     trainsform=transform
    # )
    # active_testset = NumpyDataset.buildin_dataset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_name=dataset_name,
    #     train=False,
    #     passive_feat_frac=passive_feat_frac,
    #     feat_perm_option=feat_perm_option,
    #     transform=transform
    # )

    # print('Done.')

    # 3. Initialize cryptosystem
    _crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=10,
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
    active_party = ActiveLogReg(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messenger=_messenger,
        cryptosystem=_crypto,
        logger=_logger,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        using_pool=_using_pool,
        saving_model=False,
    )

    active_party.train(active_trainset, active_testset)

    # 6. Close messenger, finish training.
    for msger_ in _messenger:
        msger_.close()

    # For online inference, you just need to substitute the model_name
    # scores = ActiveLogReg.online_inference(
    #     active_testset,
    #     model_name='20220831_185054-active_party-vertical_logreg-455_samples.model',
    #     messenger=_messenger
    # )
    #
    # print(scores)
