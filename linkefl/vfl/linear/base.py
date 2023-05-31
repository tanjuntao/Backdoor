import copy
import datetime
import multiprocessing
import multiprocessing.pool
import os
import pathlib
import time
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool

import numpy as np
from termcolor import colored

from linkefl.base import BaseMessenger, BaseModelComponent
from linkefl.common.const import Const
from linkefl.common.factory import partial_crypto_factory
from linkefl.common.log import GlobalLogger
from linkefl.crypto.paillier import encode, fast_cipher_matmul
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO


class BaseLinear(BaseModelComponent, ABC):
    def __init__(self, learning_rate, random_state):
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _init_weights(self, size):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        else:
            np.random.seed(None)  # explicitly set the seed to None
        params = np.random.normal(0, 1.0, size)
        return params

    def _gradient_descent(self, params, grad):
        params -= self.learning_rate * grad

    @abstractmethod
    def _sync_pubkey(self):
        pass

    @abstractmethod
    def _grad(self, residue, batch_idxes):
        pass

    @abstractmethod
    def train(self, trainset, validset):
        pass

    @abstractmethod
    def validate(self, validset):
        pass

    @abstractmethod
    def predict(self, testset):
        pass


class BaseLinearPassive(BaseLinear, ABC):
    def __init__(
        self,
        *,
        epochs,
        batch_size,
        learning_rate,
        messenger,
        logger,
        rank=1,
        penalty=Const.L2,
        reg_lambda=0.01,
        num_workers=1,
        val_freq=1,
        random_state=None,
        encode_precision=0.001,
        saving_model=False,
        model_dir=None,
        model_name=None,
        task="classification",
    ):
        super(BaseLinearPassive, self).__init__(learning_rate, random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.messenger = messenger
        self.logger = logger
        self.rank = rank
        self.penalty = penalty
        self.reg_lambda = reg_lambda
        self.num_workers = num_workers
        if self.num_workers > 1:
            self.executor_pool = multiprocessing.pool.ThreadPool(num_workers)
            self.scheduler_pool = multiprocessing.pool.ThreadPool(8)
        else:
            self.executor_pool = None
            self.scheduler_pool = None
        self.val_freq = val_freq
        self.random_state = random_state
        self.encode_precision = encode_precision
        self.saving_model = saving_model
        if self.saving_model:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if model_dir is None:
                default_dir = "models"
                model_dir = os.path.join(default_dir, self.create_time)
            if model_name is None:
                algo_name = (
                    Const.AlgoNames.VFL_LOGREG
                    if task == "classification"
                    else Const.AlgoNames.VFL_LINREG
                )
                model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.PASSIVE_NAME + str(self.rank),
                        algo_name=algo_name,
                    )
                    + ".model"
                )
            self.model_dir = model_dir
            self.model_name = model_name
            if not os.path.exists(self.model_dir):
                pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # given when training starts
        self.cryptosystem = None
        self.crypto_type = None

    def _sync_pubkey(self):
        self.logger.log("[PASSIVE] Requesting publie key...")
        signal = Const.START_SIGNAL
        self.messenger.send(signal)
        public_key, crypto_type = self.messenger.recv()
        self.logger.log("[PASSIVE] Done!")
        return public_key, crypto_type

    def _gradient(self, enc_residue, batch_idxes, executor_pool, scheduler_pool):
        # compute gradient of empirical loss term
        if self.crypto_type == Const.PLAIN:
            enc_train_grad = self._grad(enc_residue, batch_idxes)
        else:
            if executor_pool is None and scheduler_pool is None:
                enc_train_grad = self._grad(enc_residue, batch_idxes)
            else:
                enc_train_grad = self._grad_pool(
                    enc_residue, batch_idxes, executor_pool, scheduler_pool
                )

        # compute gradient of regularization term
        params = getattr(self, "params")
        if self.penalty == Const.NONE:
            reg_grad = np.zeros(len(params))
        elif self.penalty == Const.L1:
            reg_grad = self.reg_lambda * np.sign(params)
        elif self.penalty == Const.L2:
            reg_grad = self.reg_lambda * params
        else:
            raise ValueError("Regularization method not supported now.")
        enc_reg_grad = np.array(getattr(self, "cryptosystem").encrypt_vector(reg_grad))

        return enc_train_grad + enc_reg_grad

    def _grad(self, enc_residue, batch_idxes):
        if self.crypto_type == Const.PLAIN:
            # using x_train if crypto type is PLAIN
            enc_train_grad = -1 * (
                enc_residue[:, np.newaxis] * getattr(self, "x_train")[batch_idxes]
            ).mean(axis=0)
        else:
            # using x_encode if crypto type is Paillier or FastPaillier
            enc_train_grad = -1 * (
                enc_residue[:, np.newaxis] * getattr(self, "x_encode")[batch_idxes]
            ).mean(axis=0)

        return enc_train_grad

    def _grad_pool(self, enc_residue, batch_idxes, executor_pool, scheduler_pool):
        if self.crypto_type == Const.PLAIN:
            raise RuntimeError("you should not use pool when crypto type is Plain.")

        x_encode = getattr(self, "x_encode")
        enc_train_grad = fast_cipher_matmul(
            cipher_matrix=enc_residue[
                np.newaxis, :
            ],  # remember to add an addition axis
            plain_matrix=x_encode[batch_idxes],
            executor_pool=executor_pool,
            scheduler_pool=scheduler_pool,
        )
        # don't forget to multiply (-1/bs) to average the gradients
        enc_train_grad = (-1 / len(batch_idxes)) * enc_train_grad.flatten()

        return enc_train_grad

    @staticmethod
    def _mask_grad(enc_grad):
        perm = np.random.permutation(enc_grad.shape[0])
        return enc_grad[perm], perm

    @staticmethod
    def _unmask_grad(masked_grad, perm):
        perm_inverse = np.empty_like(perm)
        perm_inverse[perm] = np.arange(perm.size)
        true_grad = masked_grad[perm_inverse]
        return true_grad

    def train(self, trainset: NumpyDataset, validset: NumpyDataset) -> None:
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            validset, NumpyDataset
        ), "testset should be an instanceof NumpyDataset"
        setattr(self, "x_train", trainset.features)
        setattr(self, "x_val", validset.features)

        # init model parameters
        params = self._init_weights(trainset.n_features)
        setattr(self, "params", params)

        # obtain public key from active party and init cryptosystem
        public_key, crypto_type = self._sync_pubkey()
        cryptosystem = partial_crypto_factory(
            crypto_type=crypto_type,
            public_key=public_key,
            num_enc_zeros=10,
            gen_from_set=False,
        )
        self.cryptosystem = cryptosystem
        self.crypto_type = crypto_type

        # encode the training dataset if the crypto type belongs to Paillier family
        if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
            print("encoding dataset...")
            begin_time = time.time()
            x_encode = encode(
                raw_data=getattr(self, "x_train"),
                raw_pub_key=public_key,
                precision=self.encode_precision,
                num_workers=self.num_workers,
            )
            print("Done!")
            print(colored("encoding time: {}".format(time.time() - begin_time), "red"))
            setattr(self, "x_encode", x_encode)

        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1
        # Main Training Loop Here
        self.logger.log("Start collaborative model training...")
        for epoch in range(self.epochs):
            self.logger.log("Epoch: {}".format(epoch))
            all_idxes = np.arange(n_samples)
            np.random.seed(epoch)
            np.random.shuffle(all_idxes)

            for batch in range(n_batches):
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Calculate wx and send it to active party
                wx = np.matmul(
                    getattr(self, "x_train")[batch_idxes], getattr(self, "params")
                )
                self.messenger.send(wx)

                # Receive encrypted residue and calculate masked encrypted gradients
                enc_residue = self.messenger.recv()
                enc_grad = self._gradient(
                    enc_residue, batch_idxes, self.executor_pool, self.scheduler_pool
                )
                enc_mask_grad, perm = BaseLinearPassive._mask_grad(enc_grad)
                self.messenger.send(enc_mask_grad)

                # Receive decrypted masked gradient and update model
                mask_grad = self.messenger.recv()
                true_grad = BaseLinearPassive._unmask_grad(mask_grad, perm)
                self._gradient_descent(getattr(self, "params"), true_grad)

            # validate the performance of the current model
            if (epoch + 1) % self.val_freq == 0:
                self.validate(validset)
                self.validate(trainset)
                is_best = self.messenger.recv()
                if is_best:
                    print(colored("Best model updates.", "red"))
                    self.logger.log("Best model updates.")
                    if self.saving_model:
                        # the use of deepcopy here
                        # is to avoid saving other self attrbiutes
                        model_params = copy.deepcopy(getattr(self, "params"))
                        NumpyModelIO.save(model_params, self.model_dir, self.model_name)

        # after training release the multiprocessing pool
        if self.executor_pool is not None:
            self.executor_pool.close()
            self.scheduler_pool.close()
            self.executor_pool.join()
            self.scheduler_pool.join()

        if self.saving_model:
            # validate the final model
            self.validate(validset)

        self.logger.log("Finish model training.")

    def validate(self, validset: NumpyDataset) -> None:
        assert isinstance(
            validset, NumpyDataset
        ), "valset should be an instance of NumpyDataset"
        wx = np.matmul(validset.features, getattr(self, "params"))
        self.messenger.send(wx)

    def predict(self, testset: NumpyDataset) -> None:
        self.validate(testset)

    @staticmethod
    def online_inference(
        dataset: NumpyDataset,
        messenger: BaseMessenger,
        logger: GlobalLogger,
        model_dir: str,
        model_name: str,
        role: str = Const.PASSIVE_NAME,
    ):
        assert isinstance(
            dataset, NumpyDataset
        ), "inference dataset should be aninstance of NumpyDataset"
        model_params = NumpyModelIO.load(model_dir, model_name)
        wx = np.matmul(dataset.features, model_params)
        messenger.send(wx)


class BaseLinearActive(BaseLinear, ABC):
    def __init__(
        self,
        *,
        epochs,
        batch_size,
        learning_rate,
        messengers,
        cryptosystem,
        logger,
        rank=0,
        penalty=Const.L2,
        reg_lambda=0.01,
        num_workers=1,
        val_freq=1,
        random_state=None,
        saving_model=False,
        model_dir="models",
        model_name=None,
        task="classification",
    ):
        super(BaseLinearActive, self).__init__(learning_rate, random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.messengers = messengers
        self.cryptosystem = cryptosystem
        self.logger = logger
        self.rank = rank
        self.penalty = penalty
        self.reg_lambda = reg_lambda
        self.num_workers = num_workers
        if self.num_workers > 1:
            # used to accelerate encrypt_vector (for Paillier only)
            # & decrypt_vector (for both Paillier and FastPaillier)
            self.executor_pool: ThreadPool = ThreadPool(self.num_workers)
        else:
            self.executor_pool = None
        self.val_freq = val_freq
        self.random_state = random_state
        self.saving_model = saving_model
        if self.saving_model:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if model_dir is None:
                default_dir = "models"
                model_dir = os.path.join(default_dir, self.create_time)
            if model_name is None:
                algo_name = (
                    Const.AlgoNames.VFL_LOGREG
                    if task == "classification"
                    else Const.AlgoNames.VFL_LINREG
                )
                model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.ACTIVE_NAME,
                        algo_name=algo_name,
                    )
                    + ".model"
                )
            self.model_dir = model_dir
            self.model_name = model_name
            self.pics_dir = self.model_dir
            if not os.path.exists(self.model_dir):
                pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def _sync_pubkey(self):
        for msger in self.messengers:
            signal = msger.recv()
            if signal == Const.START_SIGNAL:
                print("Training protocol started.")
                print("[ACTIVE] Sending public key to passive party...")
                msger.send([self.cryptosystem.pub_key, self.cryptosystem.type])
            else:
                raise ValueError("Invalid signal, exit.")
        print("[ACTIVE] Sending public key done!")

    def _residue(self, y_true, y_hat):
        return y_true - y_hat

    def _grad(self, residue, batch_idxes):
        if not hasattr(self, "x_train"):
            raise AttributeError("x_train is not added to this object")
        if not hasattr(self, "params"):
            raise AttributeError("params is not added to this object")

        train_grad = -1 * (
            residue[:, np.newaxis] * getattr(self, "x_train")[batch_idxes]
        ).mean(axis=0)
        params = getattr(self, "params")
        if self.penalty == Const.NONE:
            reg_grad = np.zeros(len(params))
        elif self.penalty == Const.L1:
            reg_grad = self.reg_lambda * np.sign(params)
        elif self.penalty == Const.L2:
            reg_grad = self.reg_lambda * params
        else:
            raise ValueError("Regularization method not supported now.")

        return train_grad + reg_grad

    def train(self, trainset, validset):
        raise NotImplementedError("should not call this method within base class")

    def validate(self, validset):
        raise NotImplementedError("should not call this method within base class")

    def predict(self, testset):
        raise NotImplementedError("should not call this method within base class")
