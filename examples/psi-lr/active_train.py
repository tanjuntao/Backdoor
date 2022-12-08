import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
from linkefl.crypto import RSA
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale, add_intercept
from linkefl.psi.rsa import RSAPSIActive
from linkefl.vfl.linear import ActiveLogReg


if __name__ == '__main__':
    # 0. Set parameters
    trainset_path = "/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/census-active-train.csv"
    testset_path = "/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/census-active-test.csv"
    active_ip = ['localhost', 'localhost']
    active_port = [20000, 30000]
    passive_ip = ['localhost', 'localhost']
    passive_port = [20001, 30001]
    _epochs = 100
    _batch_size = 100
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.001
    _crypto_type = Const.PLAIN
    _random_state = None
    _key_size = 1024
    logger = logger_factory(role=Const.ACTIVE_NAME)

    task_start_time = time.time()

    # 1. Load dataset
    start_time = time.time()
    active_trainset = NumpyDataset.from_csv(role=Const.ACTIVE_NAME,
                                            abs_path=trainset_path,
                                            dataset_type=Const.CLASSIFICATION)
    active_testset = NumpyDataset.from_csv(role=Const.ACTIVE_NAME,
                                           abs_path=testset_path,
                                           dataset_type=Const.CLASSIFICATION)
    print(colored('1. Finish loading dataset.', 'red'))
    logger.log('1. Finish loading dataset.')
    logger.log_component(
        name=Const.DATALOADER,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0
    )

    # 2. Feature transformation
    start_time = time.time()
    active_trainset = scale(add_intercept(active_trainset))
    active_testset = scale(add_intercept(active_testset))
    print(colored('2. Finish transforming features', 'red'))
    logger.log('2. Finish transforming features')
    logger.log_component(
        name=Const.TRANSFORM,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0
    )

    # 3. Run PSI
    print(colored('3. PSI protocol started, computing...', 'red'))
    messenger = [
        messenger_factory(messenger_type=Const.FAST_SOCKET,
                          role=Const.ACTIVE_NAME,
                          active_ip=ac_ip,
                          active_port=ac_port,
                          passive_ip=pass_ip,
                          passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in
            zip(active_ip, active_port, passive_ip, passive_port)
    ]
    psi_crypto = RSA()
    active_psi = RSAPSIActive(messenger, psi_crypto, logger)
    common_ids = active_psi.run(active_trainset.ids)
    active_trainset.filter(common_ids)
    print(colored('3. Finish psi protocol', 'red'))
    logger.log('3. Finish psi protocol')

    # 4. VFL training
    print(colored('4. Training protocol started, computing...', 'red'))
    start_time = time.time()
    vfl_crypto = crypto_factory(crypto_type=_crypto_type,
                                key_size=_key_size,
                                num_enc_zeros=10000,
                                gen_from_set=False)
    active_vfl = ActiveLogReg(epochs=_epochs,
                              batch_size=_batch_size,
                              learning_rate=_learning_rate,
                              messenger=messenger,
                              cryptosystem=vfl_crypto,
                              logger=logger,
                              penalty=_penalty,
                              reg_lambda=_reg_lambda,
                              random_state=_random_state,
                              using_pool=False,
                              saving_model=True,
                              model_name='active_lr_model.model')
    active_vfl.train(active_trainset, active_testset)
    print(colored('4. Finish collaborative model training', 'red'))
    logger.log('4. Finish collaborative model training')
    logger.log_component(
        name=Const.VERTICAL_LOGREG,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0
    )

    # 5. VFL inference
    start_time = time.time()
    scores = active_vfl.predict(active_testset)
    print('Acc: {:.5f} \nAuc: {:.5f} \nf1: {:.5f}'.format(scores['acc'],
                                                          scores['auc'],
                                                          scores['f1']))
    print(colored('5. Finish collaborative inference', 'red'))
    logger.log('Acc: {:.5f} Auc: {:.5f} f1: {:.5f}'.format(scores['acc'],
                                                           scores['auc'],
                                                           scores['f1']))
    logger.log('5. Finish collaborative inference')
    logger.log_component(
        name=Const.VERTICAL_INFERENCE,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0
    )

    # 6. Finish the whole pipeline
    for msger in messenger:
        msger.close()
    print(colored('All Done.', 'red'))
    logger.log('All Done.')
    logger.log_task(
        begin=task_start_time,
        end=time.time(),
        status=Const.SUCCESS
    )


    # #For online inference, you just need to substitute the model_name
    # scores = ActiveLogReg.online_inference(
    #     active_testset,
    #     model_name='20220831_185054-active_party-vertical_logreg-455_samples.model',
    #     messenger=messenger
    # )
    #
    # print(scores)
