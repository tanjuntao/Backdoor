import copy
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from termcolor import colored

from linkefl.base import BaseCryptoSystem, BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO
from linkefl.util import sigmoid
from linkefl.vfl.linear.base import BaseLinearActive
from linkefl.vfl.utils.evaluate import Evaluate, Plot


class ActiveLogReg(BaseLinearActive, BaseModelComponent):
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
        residue_precision: float = 0.0001,
        saving_model: bool = False,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        positive_thresh: float = 0.5,
        ks_cut_points: int = 50,
    ):
        super(ActiveLogReg, self).__init__(
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
            task="classification",
        )
        self.POSITIVE_THRESH = positive_thresh
        self.RESIDUE_PRECISION = len(str(residue_precision).split(".")[1])
        self.KS_CUT_POINTS = ks_cut_points

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

    def train(
        self,
        trainset: NumpyDataset,
        validset: NumpyDataset,
    ) -> None:
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            validset, NumpyDataset
        ), "testset should be an instanceof NumpyDataset"
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

        best_acc, best_auc, best_ks = 0.0, 0.0, 0.0
        residue_records = []
        f1_records = []
        gini_records = []
        train_loss_records, valid_loss_records = [], []
        train_auc_records, valid_auc_records = [], []
        train_acc_records, valid_acc_records = [], []
        start_time = None
        # Main Training Loop Here
        self.logger.log("Start collaborative model training...")
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.logger.log("Epoch: {}".format(epoch))
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
                y_hat = sigmoid(full_wx)  # use sigmoid as activation function
                loss = self._loss(getattr(self, "y_train")[batch_idxes], y_hat)
                residue = self._residue(getattr(self, "y_train")[batch_idxes], y_hat)
                # NB: In verticalLR model, the residue (equals y_true - y_hat) may be
                # very close to zero, e.g., r = 0.000000000000...000001(50 dicimal bits)
                # then the exponent term of the encrypted residue
                # will be extreamly small,
                # e.g., -50, which will cause slow the ciphertext addition operation.
                # So you should round the residue's precision before encryption.
                if self.cryptosystem.type in (Const.PAILLIER, Const.FAST_PAILLIER):
                    residue = np.array(
                        [round(res, self.RESIDUE_PRECISION) for res in residue]
                    )
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
                train_scores = self.validate(trainset)
                self.logger.log(
                    f"Epoch: {epoch}, Train loss: {train_scores['loss']}, Valid loss:"
                    f" {scores['loss']}"
                )

                residue_records.append(np.array(batch_residues).mean())
                f1_records.append(scores["f1"])
                gini_records.append(scores["gini"])
                train_loss_records.append(train_scores["loss"])
                valid_loss_records.append(scores["loss"])
                train_auc_records.append(train_scores["auc"])
                valid_auc_records.append(scores["auc"])
                train_acc_records.append(train_scores["acc"])
                valid_acc_records.append(scores["acc"])
                if scores["acc"] > best_acc:
                    best_acc = scores["acc"]
                    is_best = True
                if scores["auc"] > best_auc:
                    best_auc = scores["auc"]
                    is_best = True
                if scores["ks"] > best_ks:
                    best_ks = scores["ks"]
                    is_best = True
                self.logger.log_metric(
                    epoch=epoch + 1,
                    loss=scores["loss"],
                    acc=scores["acc"],
                    auc=scores["auc"],
                    f1=scores["f1"],
                    ks=scores["ks"],
                    ks_threshold=scores["ks_threshold"],
                    total_epoch=self.epochs,
                )
                if is_best:
                    # save_params(self.params, role='bob')
                    print(colored("Best model updates.", "red"))
                    self.logger.log("Best model updates.")
                    if self.saving_model:
                        model_params = copy.deepcopy(getattr(self, "params"))
                        NumpyModelIO.save(model_params, self.model_dir, self.model_name)
                for msger in self.messengers:
                    msger.send(is_best)
            print(
                colored("epoch time: {}".format(time.time() - epoch_start_time), "red")
            )

        # close ThreadPool if it exists
        if self.executor_pool is not None:
            self.executor_pool.close()
            self.executor_pool.join()

        if self.saving_model:
            Plot.plot_residual(residue_records, self.pics_dir)
            Plot.plot_gini(gini_records, self.pics_dir)
            Plot.plot_f1_score(f1_records, self.pics_dir)
            Plot.plot_train_test_loss(
                train_loss_records, valid_loss_records, self.pics_dir
            )
            Plot.plot_train_test_auc(
                train_auc_records, valid_auc_records, self.pics_dir
            )
            Plot.plot_train_test_acc(
                train_acc_records, valid_acc_records, self.pics_dir
            )

            # validate the final model
            scores = self.validate(validset)
            Plot.plot_ordered_lorenz_curve(
                label=validset.labels, y_prob=scores["probs"], file_dir=self.pics_dir
            )
            Plot.plot_predict_distribution(
                y_prob=scores["probs"], bins=10, file_dir=self.pics_dir
            )
            Plot.plot_predict_prob_box(y_prob=scores["probs"], file_dir=self.pics_dir)
            Plot.plot_binary_mertics(
                validset.labels,
                scores["probs"],
                cut_point=self.KS_CUT_POINTS,
                file_dir=self.pics_dir,
            )

        self.logger.log("Finish model training.")
        self.logger.log("Best history acc: {:.5f}".format(best_acc))
        self.logger.log("Best history auc: {:.5f}".format(best_auc))
        self.logger.log("Best history ks: {:.5f}".format(best_ks))
        self.logger.log("Elapsed time: {:.5f}s".format(time.time() - start_time))
        print(colored("Best history acc: {:.5f}".format(best_acc), "red"))
        print(colored("Best history auc: {:.5f}".format(best_auc), "red"))
        print(colored("Best history ks: {:.5f}".format(best_ks), "red"))
        print(colored("Elapsed time: {:.5f}s".format(time.time() - start_time), "red"))

    def validate(self, validset: NumpyDataset) -> Dict[str, float]:
        assert isinstance(
            validset, NumpyDataset
        ), "validset should be an instance of NumpyDataset"
        active_wx = np.matmul(validset.features, getattr(self, "params"))
        full_wx = active_wx
        for msger in self.messengers:
            passive_wx = msger.recv()
            full_wx += passive_wx
        probs = sigmoid(full_wx)
        preds = (probs > self.POSITIVE_THRESH).astype(np.int32)
        loss = np.array(self._loss(validset.labels, probs)).mean()
        accuracy = accuracy_score(validset.labels, preds)
        f1 = f1_score(validset.labels, preds)
        auc = roc_auc_score(validset.labels, probs)
        gini = (auc - 0.5) / 0.5
        ks_value, ks_threshold = Evaluate.eval_ks(
            validset.labels, probs, cut_point=self.KS_CUT_POINTS
        )

        return {
            "loss": loss,
            "probs": probs,
            "preds": preds,
            "acc": accuracy,
            "f1": f1,
            "auc": auc,
            "gini": gini,
            "ks": ks_value,
            "ks_threshold": ks_threshold,
        }

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
        positive_thresh: float = 0.5,
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
        probs = sigmoid(total_wx)
        preds = (probs > positive_thresh).astype(np.int32)

        return preds


if __name__ == "__main__":
    from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
    from linkefl.feature.transform import add_intercept, parse_label, scale

    # Set parameters
    _dataset_name = "cancer"
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
    _epochs = 100
    _batch_size = 32
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.001
    _crypto_type = Const.PLAIN
    _random_state = 3347
    _key_size = 1024
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

    # Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
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

    # Initialize model and start training
    print("ACTIVE PARTY started, listening...")
    active_party = ActiveLogReg(
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
        saving_model=_saving_model,
    )
    active_party.train(active_trainset, active_testset)

    # Close messengers, finish training.
    for msger_ in _messengers:
        msger_.close()
