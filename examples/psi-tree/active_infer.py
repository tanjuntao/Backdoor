import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
from linkefl.crypto import RSA
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale, add_intercept
from linkefl.psi.rsa import RSAPSIActive
from linkefl.vfl.linear import ActiveLogReg
from linkefl.vfl.tree import ActiveTreeParty

if __name__ == '__main__':
    # 0. Set parameters
    infer_path = "census-active-test.csv"
    active_ip = ['localhost', 'localhost']
    active_port = [20000, 30000]
    passive_ip = ['localhost', 'localhost']
    passive_port = [20001, 30001]
    logger = logger_factory(role=Const.ACTIVE_NAME)

    task_start_time = time.time()

    # 1. Load dataset
    start_time = time.time()
    active_inferset = NumpyDataset.from_csv(role=Const.ACTIVE_NAME,
                                            abs_path=infer_path,
                                            dataset_type=Const.CLASSIFICATION)
    print(colored('1. Finish loading dataset.', 'red'))
    logger.log('1. Finish loading dataset.')

    # 2. Feature transformation
    active_inferset = scale(add_intercept(active_inferset))
    print(colored('2. Finish transforming features', 'red'))
    logger.log('2. Finish transforming features')

    # 3. Run PSI
    print(colored('3. PSI protocol started, computing...', 'red'))
    messengers = [
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
    active_psi = RSAPSIActive(messengers, psi_crypto, logger)
    common_ids = active_psi.run(active_inferset.ids)
    active_inferset.filter(common_ids)
    print(colored('3. Finish psi protocol', 'red'))
    logger.log('3. Finish psi protocol')

    # substitute the model_name
    scores, preds = ActiveTreeParty.online_inference(
        active_inferset, task="binary", n_labels=2, messengers=messengers, logger=logger,
        model_name='20221208_182259-active_party-vertical_sbt.model'
    )
    print(scores)
    print(preds)

    # 4. Finish the whole pipeline
    for messenger in messengers:
        messenger.close()
    print(colored('All Done.', 'red'))