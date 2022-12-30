import copy
import time
import os
import numpy as np
from sklearn.metrics import r2_score
from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory_multi
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import add_intercept, AddIntercept
from linkefl.modelio import NumpyModelIO
# from linkefl.vfl.linear import BaseLinearActive
from  linkefl.vfl.linear.base_multi import BaseLinearActive_disconnection
from  linkefl.vfl.linear.base_multi import BaseLinearPassive
from linkefl.common.factory import (
    crypto_factory,
    messenger_factory,
    partial_crypto_factory,
)

class ActiveLinReg_disconnection(BaseLinearActive_disconnection):
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
                 world_size=1,
                 reconnection=False,
                 reconnection_port=["30001"]
    ):
        super(ActiveLinReg_disconnection, self).__init__(
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
            task='regression',
            world_size=world_size,
        )
        self.reconnection = reconnection
        self.reconnection_port = reconnection_port

    def _loss(self, y_true, y_hat):
        # Linear regression uses MSE-loss as loss function
        train_loss = ((y_true - y_hat) ** 2).mean()

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

    def train(self, trainset, testset):
        assert isinstance(trainset, NumpyDataset), 'trainset should be an instance ' \
                                                   'of NumpyDataset'
        assert isinstance(testset, NumpyDataset), 'testset should be an instance ' \
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

        best_loss = float('inf')
        best_score = 0
        start_time = None
        compu_time = 0
        # Main Training Loop Here
        self.logger.log('Start collaborative model training...')
        for epoch in range(self.epochs):
            if epoch % self.val_freq == 0:
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
                active_wx = np.matmul(getattr(self, 'x_train')[batch_idxes],
                                      getattr(self, 'params'))
                full_wx = active_wx

                # for id in range(self.world_size):
                #     passive_wx = self.messenger.recv(id)
                #     full_wx += passive_wx
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
                y_hat = full_wx # no activation function
                loss = self._loss(getattr(self, 'y_train')[batch_idxes], y_hat)
                residue = self._residue(getattr(self, 'y_train')[batch_idxes], y_hat)

                # Active party helps passive party to calcalate gradient
                enc_residue = np.array(self.cryptosystem.encrypt_vector(residue))
                compu_time += time.time() - _begin

                for id in range(self.world_size):
                    passive_party[id]=self.messenger.send(enc_residue,id,passive_party[id])

                for id in range(self.world_size):
                    enc_passive_grad,passive_party[id] = self.messenger.recv(id,passive_party[id])
                    _begin = time.time()
                    if passive_party[id]:
                        passive_grad = np.array(self.cryptosystem.decrypt_vector(enc_passive_grad))
                        passive_party[id]=self.messenger.send(passive_grad,id,passive_party[id])
                    compu_time += time.time() - _begin
                # Active party calculates its gradient and update model
                active_grad = self._grad(residue, batch_idxes)
                self._gradient_descent(getattr(self, 'params'), active_grad)
                batch_losses.append(loss)

            # validate model performance
            if epoch % self.val_freq == 0:
                cur_loss = np.array(batch_losses).mean()
                self.logger.log(f"Epoch: {epoch}, Loss: {cur_loss}")
                result,output_temp_test,passive_party = self.validate(testset,output_temp_test,passive_party)
                val_loss, val_score = result['loss'], result['r2']
                if val_loss < best_loss:
                    best_loss = val_loss
                    is_best = True
                if val_score > best_score:
                    best_score = val_score
                if is_best:
                    # save_params(self.params, role='bob')
                    self.logger.log('Best model updates.')
                    if self.saving_model:
                        model_params = copy.deepcopy(getattr(self, 'params'))
                        model_name = self.model_name + "-" + str(trainset.n_samples) + "_samples" + ".model"
                        NumpyModelIO.save(model_params, self.model_path, model_name)

                # for msger in self.messenger:
                #     msger.send(is_best)
                for id in range(self.world_size): passive_party[id]=self.messenger.send(is_best,id,passive_party[id])

        self.logger.log('Finish model training.')
        self.logger.log('Best validation loss: {:.5f}'.format(best_loss))
        self.logger.log('Best r2_score: {:.5f}'.format(best_score))
        self.logger.log('Computation time: {:.5f}'.format(compu_time))
        self.logger.log('Elapsed time: {:.5f}s'.format(time.time() - start_time))
        print(colored('Best validation loss: {:.5f}'.format(best_loss), 'red'))
        print(colored('Best r2_score: {:.5f}'.format(best_score), 'red'))
        print(colored('Computation time: {:.5f}'.format(compu_time), 'red'))
        print(colored('Elapsed time: {:.5f}s'.format(time.time() - start_time), 'red'))

    def validate(self, valset,output_temp_test,passive_party):
        assert isinstance(valset, NumpyDataset), 'valset should be an instance ' \
                                                 'of NumpyDataset'
        active_wx = np.matmul(valset.features, getattr(self, 'params'))
        full_wx = active_wx
        #
        # for id in range(self.world_size ):
        #     passive_wx = self.messenger.recv(id)
        #     full_wx += passive_wx
        for id in range(self.world_size):
            passive_wx,passive_party[id] = self.messenger.recv(id,passive_party[id])
            if passive_party[id]:
                passive_data = passive_wx
                output_temp_test[id] = passive_data
            else:
                passive_data = output_temp_test[id]
            full_wx += passive_data
        y_pred = full_wx
        loss = ((valset.labels - y_pred) ** 2).mean()
        r2 = r2_score(valset.labels, y_pred)

        return {
            "loss": loss,
            "r2": r2
        }, output_temp_test,passive_party

    def predict(self, testset):
        return self.validate(testset)

    @staticmethod
    def online_inference(dataset, model_name, messenger, model_path='./models'):
        assert isinstance(dataset,
                          NumpyDataset), 'inference dataset should be an ' \
                                         'instance of NumpyDataset'
        model_params = NumpyModelIO.load(model_path, model_name)
        active_wx = np.matmul(dataset.features, model_params)
        passive_wx = messenger.recv()
        y_pred = active_wx + passive_wx
        loss = ((dataset.labels - y_pred) ** 2).mean()
        r2 = r2_score(dataset.labels, y_pred)
        scores = {
            "loss": loss,
            "r2": r2
        }
        messenger.send(scores)

        return scores



class PassiveLinReg_disconnection(BaseLinearPassive):
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
        super(PassiveLinReg_disconnection, self).__init__(
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

class PassiveLinReg_reconnection(BaseLinearPassive):
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
                 model_path='./models',
    ):
        super(PassiveLinReg_reconnection, self).__init__(
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
        print(new_epoch)
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
