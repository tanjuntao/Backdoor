import copy
import time

# from line_profiler_pycharm import profile
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import (
    crypto_factory,
    logger_factory,
    messenger_factory,
    messenger_factory_multi,
)
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import (
    AddIntercept,
    Compose,
    ParseLabel,
    Scale,
    add_intercept,
    parse_label,
    scale,
)
from linkefl.modelio import NumpyModelIO
from linkefl.util import save_params, sigmoid

# from linkefl.vfl.linear import BaseLinearActive
from linkefl.vfl.linear.base_multi import BaseLinearActive


class ActiveLogReg(BaseLinearActive):
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
        positive_thresh=0.5,
        residue_precision=0.0001,
        world_size=1,
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
            task="classification",
            world_size=world_size,
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
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.logger.log("Epoch: {}".format(epoch))
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

                for id in range(self.world_size):
                    passive_wx = self.messenger.recv(id)
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

                for id in range(self.world_size):
                    self.messenger.send(enc_residue, id)

                for id in range(self.world_size):
                    enc_passive_grad = self.messenger.recv(id)
                    _begin = time.time()
                    passive_grad = np.array(
                        self.cryptosystem.decrypt_vector(enc_passive_grad)
                    )
                    compu_time += time.time() - _begin
                    self.messenger.send(passive_grad, id)

                # Active party calculates its gradient and update model
                active_grad = self._grad(residue, batch_idxes)
                self._gradient_descent(getattr(self, "params"), active_grad)
                batch_losses.append(loss)

            # validate model performance
            if epoch % self.val_freq == 0:
                cur_loss = np.array(batch_losses).mean()
                self.logger.log(f"Epoch: {epoch}, Loss: {cur_loss}")

                scores = self.validate(testset)
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
                if is_best:
                    # save_params(self.params, role='bob')
                    self.logger.log("Best model updates.")
                    if self.saving_model:
                        model_params = copy.deepcopy(getattr(self, "params"))
                        model_name = (
                            self.model_name
                            + "-"
                            + str(trainset.n_samples)
                            + "_samples"
                            + ".model"
                        )
                        NumpyModelIO.save(model_params, self.model_path, model_name)
                for id in range(self.world_size):
                    self.messenger.send(is_best, id)
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

    def validate(self, valset):
        assert isinstance(
            valset, NumpyDataset
        ), "valset should be an instance of NumpyDataset"
        active_wx = np.matmul(valset.features, getattr(self, "params"))
        full_wx = active_wx

        # for msger in self.messenger:
        #     passive_wx = msger.recv()
        #     full_wx += passive_wx
        for id in range(self.world_size):
            passive_wx = self.messenger.recv(id)
            full_wx += passive_wx
        probs = sigmoid(full_wx)
        preds = (probs > self.POSITIVE_THRESH).astype(np.int32)

        accuracy = accuracy_score(valset.labels, preds)
        f1 = f1_score(valset.labels, preds)
        auc = roc_auc_score(valset.labels, probs)

        return {"acc": accuracy, "f1": f1, "auc": auc}

    def predict(self, testset):
        return self.validate(testset)

    @staticmethod
    def online_inference(
        dataset, model_name, messenger, model_path="./models", POSITIVE_THRESH=0.5
    ):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be an instance of NumpyDataset"
        model_params = NumpyModelIO.load(model_path, model_name)
        active_wx = np.matmul(dataset.features, model_params)
        passive_wx = messenger.recv()
        probs = sigmoid(active_wx + passive_wx)
        preds = (probs > POSITIVE_THRESH).astype(np.int32)
        accuracy = accuracy_score(dataset.labels, preds)
        f1 = f1_score(dataset.labels, preds)
        auc = roc_auc_score(dataset.labels, probs)

        scores = {"acc": accuracy, "auc": auc, "f1": f1}
        messenger.send(scores)
        return scores


if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "census"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = ["localhost", "localhost"]
    active_port = [20000, 30000]
    passive_ip = ["localhost", "localhost"]
    passive_port = [20001, 30001]
    world_size = 2
    _epochs = 100
    _batch_size = 100
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
    # _messenger = [
    #     messenger_factory(messenger_type=Const.FAST_SOCKET,
    #                       role=Const.ACTIVE_NAME,
    #                       active_ip=ac_ip,
    #                       active_port=ac_port,
    #                       passive_ip=pass_ip,
    #                       passive_port=pass_port,
    #     )
    #     for ac_ip, ac_port, pass_ip, pass_port in
    #         zip(active_ip, active_port, passive_ip, passive_port)
    # ]
    _messenger = messenger_factory_multi(
        messenger_type=Const.FAST_SOCKET,
        role=Const.ACTIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
        world_size=world_size,
    )

    print("ACTIVE PARTY started, connecting...")

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
        world_size=world_size,
    )

    active_party.train(active_trainset, active_testset)

    # 6. Close messenger, finish training.
    # for msger_ in _messenger:
    #     msger_.close()
    _messenger.close()

    # For online inference, you just need to substitute the model_name
    # scores = ActiveLogReg.online_inference(
    #     active_testset,
    #     model_name='20220831_185054-active_party-vertical_logreg-455_samples.model',
    #     messenger=_messenger
    # )
    #
    # print(scores)
