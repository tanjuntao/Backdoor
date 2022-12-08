import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import RSAPSIPassive
from linkefl.vfl.linear import PassiveLogReg
from linkefl.vfl.tree import PassiveTreeParty

if __name__ == '__main__':
    # 0. Set parameters
    inferset_path = "census-passive1-test.csv"
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 20001
    logger = logger_factory(role=Const.PASSIVE_NAME)

    # 1. Load dataset
    passive_inferset = NumpyDataset.from_csv(role=Const.PASSIVE_NAME,
                                             abs_path=inferset_path,
                                             dataset_type=Const.CLASSIFICATION)
    print(colored('1. Finish loading dataset.', 'red'))
    logger.log('1. Finish loading dataset.')

    # 2. Feature transformation
    passive_inferset = scale(passive_inferset)
    print(colored('2. Finish transforming features', 'red'))
    logger.log('2. Finish transforming features')

    # 3. Run PSI
    print(colored('3. PSI protocol started, computing...', 'red'))
    messenger = FastSocket(role=Const.PASSIVE_NAME,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port)
    passive_psi = RSAPSIPassive(messenger, logger)
    common_ids = passive_psi.run(passive_inferset.ids)
    passive_inferset.filter(common_ids)
    print(colored('3. Finish psi protocol', 'red'))
    logger.log('3. Finish psi protocol')

    # For online inference, you just need to substitute the model_name
    scores, preds = PassiveTreeParty.online_inference(
        passive_inferset, messenger, logger,
        model_name="20221208_182250-passive_party-vertical_sbt-32561_samples.model"
    )
    print(scores)
    print(preds)

    # 6. Finish the whole pipeline
    messenger.close()
    print(colored('All Done.', 'red'))
