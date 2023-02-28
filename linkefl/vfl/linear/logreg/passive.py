from linkefl.base import BaseModelComponent
from linkefl.common.const import Const
from linkefl.vfl.linear import BaseLinearPassive

import copy
import datetime
import multiprocessing
import multiprocessing.pool
import os
import time
from abc import ABC, abstractmethod

import numpy as np

from linkefl.common.const import Const
from linkefl.common.factory import (
    crypto_factory,
    messenger_factory,
    partial_crypto_factory,
)
from linkefl.config import BaseConfig
from linkefl.crypto.paillier import cipher_matmul, encode
from linkefl.dataio import NumpyDataset
from linkefl.modelio import NumpyModelIO


class PassiveLogReg(BaseLinearPassive, BaseModelComponent):
    def __init__(
        self,
        epochs,
        batch_size,
        learning_rate,
        messenger,
        crypto_type,
        logger,
        *,
        rank=1,
        penalty=Const.L2,
        reg_lambda=0.01,
        precision=0.001,
        random_state=None,
        using_pool=False,
        num_workers=-1,
        val_freq=1,
        saving_model=False,
        model_path="./models",
        model_name=None,
    ):
        super(PassiveLogReg, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messenger=messenger,
            crypto_type=crypto_type,
            logger=logger,
            rank=rank,
            penalty=penalty,
            reg_lambda=reg_lambda,
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

    def fit(self, trainset, validset, role=Const.PASSIVE_NAME):
        self.train(trainset, validset)

    def train(self, trainset, testset):
        assert isinstance(
            trainset, NumpyDataset
        ), "trainset should be an instance of NumpyDataset"
        assert isinstance(
            testset, NumpyDataset
        ), "testset should be an instanceof NumpyDataset"
        setattr(self, "x_train", trainset.features)
        setattr(self, "x_val", testset.features)
        # init model parameters
        params = self._init_weights(trainset.n_features)
        setattr(self, "params", params)

        # obtain public key from active party and init cryptosystem
        public_key = self._sync_pubkey()
        cryptosystem = partial_crypto_factory(
            crypto_type=self.crypto_type,
            public_key=public_key,
            num_enc_zeros=10,
            gen_from_set=False,
        )
        setattr(self, "cryptosystem", cryptosystem)

        # encode the training dataset if the crypto type belongs to Paillier family
        if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
            print("encoding dataset...")
            x_encode = encode(
                raw_data=getattr(self, "x_train"),
                raw_pub_key=public_key,
                precision=self.precision,
            )
            print("Done!")
            setattr(self, "x_encode", x_encode)

        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        commu_plus_compu_time = 0
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
                _begin = time.time()
                self.messenger.send(wx)

                # Receive encrypted residue and calculate masked encrypted gradients
                enc_residue = self.messenger.recv()
                commu_plus_compu_time += time.time() - _begin
                enc_grad = self._gradient(
                    enc_residue, batch_idxes, self.executor_pool, self.scheduler_pool
                )
                enc_mask_grad, perm = BaseLinearPassive._mask_grad(enc_grad)
                _begin = time.time()
                self.messenger.send(enc_mask_grad)

                # Receive decrypted masked gradient and update model
                mask_grad = self.messenger.recv()
                true_grad = BaseLinearPassive._unmask_grad(mask_grad, perm)
                commu_plus_compu_time += time.time() - _begin
                self._gradient_descent(getattr(self, "params"), true_grad)

            # validate the performance of the current model
            if epoch % self.val_freq == 0:
                self.validate(testset)
                self.validate(trainset)     # to get evaluate message
                is_best = self.messenger.recv()
                if is_best:
                    # save_params(self.params, role=Const.PASSIVE_NAME)
                    self.logger.log("Best model updates.")
                    if self.saving_model:
                        # the use of deepcopy here
                        # is to avoid saving other self attrbiutes
                        model_params = copy.deepcopy(getattr(self, "params"))
                        NumpyModelIO.save(
                            model_params, self.model_path, self.model_name
                        )

        # after training release the multiprocessing pool
        if self.executor_pool is not None:
            self.executor_pool.close()
            self.scheduler_pool.close()
            self.executor_pool.join()
            self.scheduler_pool.join()

        self.validate(testset)  # for evaluate message
        self.logger.log("Finish model training.")
        self.logger.log(
            "The summation of communication time and computation time is "
            "{:.4f}s. In order to obtain the communication time only, you should"
            "substract the computation time which is printed in the console"
            "of active party.".format(commu_plus_compu_time)
        )

    def score(self, testset, role=Const.PASSIVE_NAME):
        return self.predict(testset)


if __name__ == "__main__":
    from linkefl.common.factory import logger_factory, messenger_factory
    from linkefl.dataio import NumpyDataset
    from linkefl.feature.transform import scale

    # 0. Set parameters
    _dataset_name = "cancer"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = "localhost"
    active_port = 20000
    passive_ip = "localhost"
    passive_port = 20001
    _epochs = 100
    _batch_size = 32
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.01
    _random_state = 3347
    _crypto_type = Const.PLAIN
    _using_pool = False

    # 1. Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print("Loading dataset...")
    passive_trainset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_testset = NumpyDataset.buildin_dataset(
        role=Const.PASSIVE_NAME,
        dataset_name=_dataset_name,
        root="../../data",
        train=False,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_trainset = NumpyDataset.feature_split(passive_trainset, n_splits=2)[0]
    passive_testset = NumpyDataset.feature_split(passive_testset, n_splits=2)[0]
    # load dummy dataset
    # dummy_dataset = NumpyDataset.dummy_daaset(
    #     role=Const.PASSIVE_NAME,
    #     dataset_type=Const.CLASSIFICATION,
    #     n_samples=100000,
    #     n_features=100,
    #     passive_feat_frac=passive_feat_frac
    # )
    # passive_trainset, passive_testset = NumpyDataset.train_test_split(
    #     dummy_dataset,
    #     test_size=0.2
    # )
    passive_trainset = scale(passive_trainset)
    passive_testset = scale(passive_testset)

    # Option 2: PyTorch style
    # print('Loading dataset...')
    # transform = Scale()
    # passive_trainset = NumpyDataset.buildin_dataset(
    #     role=Const.PASSIVE_NAME,
    #     dataset_name=dataset_name,
    #     train=True,
    #     passive_feat_frac=passive_feat_frac,
    #     feat_perm_option=feat_perm_option,
    #     transform=transform
    # )
    # passive_testset = NumpyDataset.buildin_dataset(
    #     role=Const.PASSIVE_NAME,
    #     dataset_name=dataset_name,
    #     train=False,
    #     passive_feat_frac=passive_feat_frac,
    #     feat_perm_option=feat_perm_option,
    #     transform=transform
    # )
    # print('Done.')

    # 3. Initialize messenger
    _messenger = messenger_factory(
        messenger_type=Const.FAST_SOCKET,
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    # 4. Initialize model and start training
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    passive_party = PassiveLogReg(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messenger=_messenger,
        crypto_type=_crypto_type,
        logger=_logger,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        using_pool=_using_pool,
        saving_model=False,
    )

    passive_party.train(passive_trainset, passive_testset)

    # 5. Close messenger, finish training
    _messenger.close()

    # For online inference, you just need to substitute the model_name
    # scores = PassiveLogReg.online_inference(
    #     passive_testset,
    #     model_name='20220831_185109-passive_party-vertical_logreg-455_samples.model',
    #     messenger=_messenger
    # )
    # print(scores)
