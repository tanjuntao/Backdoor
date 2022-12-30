import numpy as np
import copy
import time
from termcolor import colored

from linkefl.util import sigmoid, save_params
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score
from linkefl.common.const import Const
from linkefl.common.factory import logger_factory, messenger_factory,messenger_factory_multi_disconnection
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale, Scale
# from linkefl.vfl.linear import BaseLinearPassive
from  linkefl.vfl.linear.base_multi import BaseLinearPassive
from linkefl.modelio import NumpyModelIO
from linkefl.common.factory import (
    crypto_factory,
    messenger_factory,
    partial_crypto_factory,
)
from  linkefl.vfl.linear.base_multi import BaseLinearActive_disconnection
import os


class PassiveLogReg_disconnection(BaseLinearPassive):
    def __init__(self,
                 epochs,
                 batch_size,
                 learning_rate,
                 messenger,
                 crypto_type,
                 logger,
                 *,
                 penalty=Const.L2,
                 reg_lambda=0.01,
                 precision=0.001,
                 random_state=None,
                 using_pool=False,
                 num_workers=-1,
                 val_freq=1,
                 saving_model=False,
                 model_path='./models_passive',
    ):
        super(PassiveLogReg_disconnection, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messenger=messenger,
            crypto_type=crypto_type,
            logger=logger,
            penalty=penalty,
            reg_lambda=reg_lambda,
            precision=precision,
            random_state=random_state,
            using_pool=using_pool,
            num_workers=num_workers,
            val_freq=val_freq,
            saving_model=saving_model,
            model_path=model_path,
            task='classification'
        )


    def send_init_output(self,n_batches,all_idxes,bs,valset):
        init_output = []
        for batch in range(n_batches):
            # Choose batch indexes
            start = batch * bs
            end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
            batch_idxes = all_idxes[start:end]
            # Calculate wx and send it to active party
            wx = np.matmul(getattr(self, 'x_train')[batch_idxes], getattr(self, 'params'))
            init_output.append(wx)
        wx = np.matmul(valset.features, getattr(self, 'params'))
        init_output.append(wx)
        self.messenger.send(init_output)

    def train(self, trainset, testset):
        assert isinstance(trainset, NumpyDataset), 'trainset should be an instance ' \
                                                   'of NumpyDataset'
        assert isinstance(testset, NumpyDataset), 'testset should be an instance' \
                                                  'of NumpyDataset'
        setattr(self, 'x_train', trainset.features)
        setattr(self, 'x_val', testset.features)
        # init model parameters
        params = self._init_weights(trainset.n_features)
        setattr(self, 'params', params)

        # obtain public key from active party and init cryptosystem
        public_key = self._sync_pubkey()
        cryptosystem = partial_crypto_factory(crypto_type=self.crypto_type,
                                              public_key=public_key,
                                              num_enc_zeros=10,
                                              gen_from_set=False)
        setattr(self, 'cryptosystem', cryptosystem)

        # encode the training dataset if the crypto type belongs to Paillier family
        if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
            print('encoding dataset...')
            x_encode = BaseLinearPassive._encode(
                getattr(self, 'x_train'),
                public_key,
                self.precision
            )
            print('Done!')
            setattr(self, 'x_encode', x_encode)


        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        commu_plus_compu_time = 0
        # Main Training Loop Here
        all_idxes = list(range(n_samples))

        self.send_init_output(n_batches, all_idxes, bs,testset)

        self.logger.log('Start collaborative model training...')
        for epoch in range(self.epochs):
            self.logger.log('Epoch: {}'.format(epoch))
            all_idxes = np.arange(n_samples)
            np.random.seed(epoch)
            np.random.shuffle(all_idxes)

            for batch in range(n_batches):
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Calculate wx and send it to active party
                wx = np.matmul(getattr(self, 'x_train')[batch_idxes], getattr(self, 'params'))

                _begin = time.time()
                self.messenger.send(wx)

                # Receive encrypted residue and calculate masked encrypted gradients
                enc_residue = self.messenger.recv()
                commu_plus_compu_time += time.time() - _begin
                enc_grad = self._gradient(
                    enc_residue,
                    batch_idxes,
                    self.executor_pool,
                    self.scheduler_pool
                )
                enc_mask_grad, perm = BaseLinearPassive._mask_grad(enc_grad)
                _begin = time.time()
                self.messenger.send(enc_mask_grad)

                # Receive decrypted masked gradient and update model
                mask_grad = self.messenger.recv()
                true_grad = BaseLinearPassive._unmask_grad(mask_grad, perm)
                commu_plus_compu_time += time.time() - _begin
                self._gradient_descent(getattr(self, 'params'), true_grad)

            # validate the performance of the current model
            if epoch % self.val_freq == 0:
                self.validate(testset)
                is_best = self.messenger.recv()
                if is_best:
                    # save_params(self.params, role=Const.PASSIVE_NAME)
                    self.logger.log('Best model updates.')
                    if self.saving_model:
                        # the use of deepcopy here is to avoid saving other self attrbiutes
                        model_params = copy.deepcopy(getattr(self, 'params'))
                        model_name = self.model_name + "-" + str(trainset.n_samples) + "_samples" + ".model"
                        NumpyModelIO.save(model_params, self.model_path, model_name)

        # after training release the multiprocessing pool
        if self.executor_pool is not None:
            self.executor_pool.close()
            self.scheduler_pool.close()
            self.executor_pool.join()
            self.scheduler_pool.join()

        self.logger.log('Finish model training.')
        self.logger.log('The summation of communication time and computation time is '
              '{:.4f}s. In order to obtain the communication time only, you should'
              'substract the computation time which is printed in the console'
              'of active party.'.format(commu_plus_compu_time))

class PassiveLogReg_reconnection(BaseLinearPassive):
    def __init__(self,
                 epochs,
                 batch_size,
                 learning_rate,
                 messenger,
                 crypto_type,
                 logger,
                 *,
                 penalty=Const.L2,
                 reg_lambda=0.01,
                 precision=0.001,
                 random_state=None,
                 using_pool=False,
                 num_workers=-1,
                 val_freq=1,
                 saving_model=False,
                 model_path='./models_passive',
    ):
        super(PassiveLogReg_reconnection, self).__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            messenger=messenger,
            crypto_type=crypto_type,
            logger=logger,
            penalty=penalty,
            reg_lambda=reg_lambda,
            precision=precision,
            random_state=random_state,
            using_pool=using_pool,
            num_workers=num_workers,
            val_freq=val_freq,
            saving_model=saving_model,
            model_path=model_path,
            task='classification'
        )

    def train(self, trainset, testset):
        assert isinstance(trainset, NumpyDataset), 'trainset should be an instance ' \
                                                   'of NumpyDataset'
        assert isinstance(testset, NumpyDataset), 'testset should be an instance' \
                                                  'of NumpyDataset'
        setattr(self, 'x_train', trainset.features)
        setattr(self, 'x_val', testset.features)


        # obtain public key from active party and init cryptosystem
        public_key = self._sync_pubkey()
        cryptosystem = partial_crypto_factory(crypto_type=self.crypto_type,
                                              public_key=public_key,
                                              num_enc_zeros=10,
                                              gen_from_set=False)
        setattr(self, 'cryptosystem', cryptosystem)

        # encode the training dataset if the crypto type belongs to Paillier family
        if self.crypto_type in (Const.PAILLIER, Const.FAST_PAILLIER):
            print('encoding dataset...')
            x_encode = BaseLinearPassive._encode(
                getattr(self, 'x_train'),
                public_key,
                self.precision
            )
            print('Done!')
            setattr(self, 'x_encode', x_encode)


        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        commu_plus_compu_time = 0
        # Main Training Loop Here
        all_idxes = list(range(n_samples))
        new_epoch = self.messenger.recv()
        self.logger.log('Start collaborative model training...')
        for epoch in range(new_epoch, self.epochs):
            self.logger.log('Epoch: {}'.format(epoch))
            all_idxes = np.arange(n_samples)
            np.random.seed(epoch)
            np.random.shuffle(all_idxes)

            for batch in range(n_batches):
                # Choose batch indexes
                start = batch * bs
                end = len(all_idxes) if batch == n_batches - 1 else (batch + 1) * bs
                batch_idxes = all_idxes[start:end]

                # Calculate wx and send it to active party
                wx = np.matmul(getattr(self, 'x_train')[batch_idxes], getattr(self, 'params'))

                _begin = time.time()
                self.messenger.send(wx)

                # Receive encrypted residue and calculate masked encrypted gradients
                enc_residue = self.messenger.recv()
                commu_plus_compu_time += time.time() - _begin
                enc_grad = self._gradient(
                    enc_residue,
                    batch_idxes,
                    self.executor_pool,
                    self.scheduler_pool
                )
                enc_mask_grad, perm = BaseLinearPassive._mask_grad(enc_grad)
                _begin = time.time()
                self.messenger.send(enc_mask_grad)

                # Receive decrypted masked gradient and update model
                mask_grad = self.messenger.recv()
                true_grad = BaseLinearPassive._unmask_grad(mask_grad, perm)
                commu_plus_compu_time += time.time() - _begin
                self._gradient_descent(getattr(self, 'params'), true_grad)

            # validate the performance of the current model
            if epoch % self.val_freq == 0:
                self.validate(testset)
                is_best = self.messenger.recv()
                if is_best:
                    # save_params(self.params, role=Const.PASSIVE_NAME)
                    self.logger.log('Best model updates.')
                    if self.saving_model:
                        # the use of deepcopy here is to avoid saving other self attrbiutes
                        model_params = copy.deepcopy(getattr(self, 'params'))
                        model_name = self.model_name + "-" + str(trainset.n_samples) + "_samples" + ".model"
                        NumpyModelIO.save(model_params, self.model_path, model_name)

        # after training release the multiprocessing pool
        if self.executor_pool is not None:
            self.executor_pool.close()
            self.scheduler_pool.close()
            self.executor_pool.join()
            self.scheduler_pool.join()

        self.logger.log('Finish model training.')
        self.logger.log('The summation of communication time and computation time is '
              '{:.4f}s. In order to obtain the communication time only, you should'
              'substract the computation time which is printed in the console'
              'of active party.'.format(commu_plus_compu_time))
        
    def get_latest_filename(self,filedir):
        if os.path.exists(filedir):
            file_list = os.listdir(filedir)
        else:
            raise ValueError("not exist filedir.")

        # sort by create time
        file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(filedir, fn)))
        return file_list[-1]

    def load_lastmodel(self,model_path):

        model_name = self.get_latest_filename(model_path)
        model_params = NumpyModelIO.load(model_path, model_name)

        setattr(self, 'params', model_params)

class ActiveLogReg_disconnection(BaseLinearActive_disconnection):
    def __init__(self,
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
                 model_path='./models_active',
                 positive_thresh=0.5,
                 residue_precision=0.0001,
                 world_size=1,
                 reconnection=False,
                 reconnection_port=["30001"]

    ):
        super(ActiveLogReg_disconnection, self).__init__(
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
            task='classification',
            world_size=world_size
        )
        self.POSITIVE_THRESH = positive_thresh
        self.RESIDUE_PRECISION = len(str(residue_precision).split('.')[1])
        self.reconnection = reconnection
        self.reconnection_port = reconnection_port
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
        train_loss = ActiveLogReg_disconnection._logloss(y_true, y_hat)

        params = getattr(self, 'params')
        if self.penalty == Const.NONE:
            reg_loss = 0.0
        elif self.penalty == Const.L1:
            reg_loss = self.reg_lambda * abs(params).sum()
        elif self.penalty == Const.L2:
            reg_loss = 1. / 2 * self.reg_lambda * (params ** 2).sum()
        else:
            raise ValueError('Regularization method not supported now.')
        total_loss = train_loss + reg_loss

        return total_loss

    def get_init_output(self,passive_party):
        init_output = []
        for id in range(self.world_size):
            temp,passive_party[id]=self.messenger.recv(id,passive_party[id])
            init_output.append(temp)
        return init_output,passive_party

    def try_to_connection(self,id,reconnection_port):
        print(colored("Try To Connect {} ",'red').format(id))
        party = self.messenger.try_reconnect(reconnection_port=reconnection_port,id=id)
        if party:
            try:
                self._sync_pubkey_client(id)
                party = True
            except Exception:
                party = False
        if party:
            print(colored("Reconnection Successfully!!!", 'red'))
        else:
            print(colored("Reconnection Failed!!!", 'red'))
        return party

    # @profile
    def train(self, trainset, testset):
        assert isinstance(trainset, NumpyDataset), 'trainset should be an instance ' \
                                                   'of NumpyDataset'
        assert isinstance(testset, NumpyDataset), 'testset should be an instance' \
                                                  'of NumpyDataset'
        setattr(self, 'x_train', trainset.features)
        setattr(self, 'x_val', testset.features)
        setattr(self, 'y_train', trainset.labels)
        setattr(self, 'y_val', testset.labels)

        # initialize model parameters
        params = self._init_weights(trainset.n_features)
        setattr(self, 'params', params)

        # trainfer public key to passive party
        self._sync_pubkey()

        bs = self.batch_size if self.batch_size != -1 else trainset.n_samples
        n_samples = trainset.n_samples
        if n_samples % bs == 0:
            n_batches = n_samples // bs
        else:
            n_batches = n_samples // bs + 1

        passive_party = [True for i in range(self.world_size)]
        init_output,passive_party = self.get_init_output(passive_party)



        passive_party_all = True
        for i in range(self.world_size):
            passive_party_all = passive_party_all*passive_party[i]

        if passive_party_all:
            print("[ACTIVE] Init Done!")
        else:
            print(colored("Initialization failed, please restart training!!!",'red'))
            return 0

        out_temp = []
        output_temp_test = []
        for i in range(self.world_size):
            t = init_output[i]
            out_temp.append([t[j] for j in range(len(t)-1)])
            output_temp_test.append(t[-1])


        best_acc, best_auc = 0.0, 0.0
        start_time = None
        compu_time = 0
        # Main Training Loop Here



        self.logger.log('Start collaborative model training...')
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.logger.log('Epoch: {}'.format(epoch))

            if self.reconnection:
                for id in range(self.world_size):
                    if not passive_party[id]:
                        passive_party[id] = self.try_to_connection(id=id,reconnection_port=self.reconnection_port)
                        passive_party[id] = self.messenger.send(epoch,id,passive_party[id])


            for id in range(self.world_size):
                if passive_party[id]:
                    print(colored('passive_party {} status: connection', 'red').format(id))
                else:
                    print(colored('passive_party {} status: disconnection', 'red').format(id))

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
                active_wx = np.matmul(getattr(self, 'x_train')[batch_idxes], getattr(self, 'params'))
                full_wx = active_wx

                for id in range(self.world_size):
                    passive_wx,passive_party[id] = self.messenger.recv(id,passive_party[id])
                    if passive_party[id]:
                        passive_data = passive_wx
                        out_temp[id][batch] = passive_data
                    else:
                        passive_data = out_temp[id][batch]
                    full_wx += passive_data

                _begin = time.time()
                if start_time is None:
                    start_time = time.time()
                y_hat = sigmoid(full_wx) # use sigmoid as activation function
                loss = self._loss(getattr(self, 'y_train')[batch_idxes], y_hat)
                residue = self._residue(getattr(self, 'y_train')[batch_idxes], y_hat)
                # NB: In verticalLR model, the residue (equals y_true - y_hat) may be
                # very close to zero, e.g., r = 0.000000000000...000001 (50 dicimal bits)
                # then the exponent term of the encrypted residue will be extreamly small,
                # e.g., -50, which will cause slow the ciphertext addition operation.
                # So you should round the residue's precision before encryption.
                if self.using_pool:
                    residue = np.array([round(res, self.RESIDUE_PRECISION) for res in residue])

                # Active party helps passive party to calcalate gradient
                enc_residue = np.array(self.cryptosystem.encrypt_vector(residue))

                for id in range(self.world_size):
                    passive_party[id] = self.messenger.send(enc_residue,id,passive_party[id])

                for id in range(self.world_size):
                    enc_passive_grad,passive_party[id] = self.messenger.recv(id,passive_party[id])
                    _begin = time.time()
                    if passive_party[id]:
                        passive_grad = np.array(self.cryptosystem.decrypt_vector(enc_passive_grad))
                        passive_party[id] = self.messenger.send(passive_grad,id,passive_party[id])
                    compu_time += time.time() - _begin

                # Active party calculates its gradient and update model
                active_grad = self._grad(residue, batch_idxes)
                self._gradient_descent(getattr(self, 'params'), active_grad)
                batch_losses.append(loss)

            # validate model performance
            if epoch % self.val_freq == 0:
                cur_loss = np.array(batch_losses).mean()
                self.logger.log(f"Epoch: {epoch}, Loss: {cur_loss}")

                scores,passive_party,output_temp_test = self.validate(testset,output_temp_test=output_temp_test,
                                                                        passive_party=passive_party)
                if scores['acc'] > best_acc:
                    best_acc = scores['acc']
                    is_best = True
                if scores['auc'] > best_auc:
                    best_auc = scores['auc']
                    is_best = True
                self.logger.log_metric(epoch,
                                       cur_loss,
                                       scores['acc'], scores['auc'], scores['f1'],
                                       total_epoch=self.epochs)
                if is_best:
                    # save_params(self.params, role='bob')
                    self.logger.log('Best model updates.')
                    if self.saving_model:
                        model_params = copy.deepcopy(getattr(self, 'params'))
                        model_name = self.model_name + "-" + str(trainset.n_samples) + "_samples" + ".model"
                        NumpyModelIO.save(model_params, self.model_path, model_name)
                for id in range(self.world_size):
                    passive_party[id] = self.messenger.send(is_best,id,passive_party[id])
            print(colored('epoch time: {}'.format(time.time() - epoch_start_time), 'red'))

        # close ThreadPool if it exists
        if self.executor_pool is not None:
            self.executor_pool.close()
            self.executor_pool.join()

        self.logger.log('Finish model training.')
        self.logger.log('Best history acc: {:.5f}'.format(best_acc))
        self.logger.log('Best history auc: {:.5f}'.format(best_auc))
        self.logger.log('Computation time: {:.5f}'.format(compu_time))
        self.logger.log('Elapsed time: {:.5f}s'.format(time.time() - start_time))
        print(colored('Best history acc: {:.5f}'.format(best_acc), 'red'))
        print(colored('Best history auc: {:.5f}'.format(best_auc), 'red'))
        print(colored('Computation time: {:.5f}'.format(compu_time), 'red'))
        print(colored('Elapsed time: {:.5f}s'.format(time.time() - start_time), 'red'))

    def validate(self, valset,output_temp_test,passive_party):
        assert isinstance(valset, NumpyDataset), 'valset should be an instance ' \
                                                 'of NumpyDataset'
        active_wx = np.matmul(valset.features, getattr(self, 'params'))
        full_wx = active_wx

        for id in range(self.world_size):
            passive_wx,passive_party[id] = self.messenger.recv(id,passive_party[id])
            if passive_party[id]:
                passive_data = passive_wx
                output_temp_test[id] = passive_data
            else:
                passive_data = output_temp_test[id]
            full_wx += passive_data

        probs = sigmoid(full_wx)
        preds = (probs > self.POSITIVE_THRESH).astype(np.int32)

        accuracy = accuracy_score(valset.labels, preds)
        f1 = f1_score(valset.labels, preds)
        auc = roc_auc_score(valset.labels, probs)

        return {
            'acc': accuracy,
            'f1': f1,
            'auc': auc
        },passive_party, output_temp_test

    def predict(self, testset):
        return self.validate(testset)

    @staticmethod
    def online_inference(dataset, model_name, messenger,
                         model_path='./models', POSITIVE_THRESH=0.5):
        assert isinstance(dataset, NumpyDataset), 'inference dataset should be an ' \
                                                  'instance of NumpyDataset'
        model_params = NumpyModelIO.load(model_path, model_name)
        active_wx = np.matmul(dataset.features, model_params)
        passive_wx = messenger.recv()
        probs = sigmoid(active_wx + passive_wx)
        preds = (probs > POSITIVE_THRESH).astype(np.int32)
        accuracy = accuracy_score(dataset.labels, preds)
        f1 = f1_score(dataset.labels, preds)
        auc = roc_auc_score(dataset.labels, probs)

        scores = {
            "acc": accuracy,
            "auc": auc,
            "f1": f1
        }
        messenger.send(scores)
        return scores

