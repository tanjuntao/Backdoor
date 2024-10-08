import os
import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import PassiveRSAPSI
from linkefl.vfl.linear import PassiveLogReg

if __name__ == "__main__":
    # 0. Set parameters
    db_host = "localhost"
    db_user = "tiger"
    db_name = "hello_db"
    db_table_name = "hello_table"
    db_password = "hello_pw"
    active_ip = "localhost"
    active_port = 20000
    passive_ip = "localhost"
    passive_port = 20001
    logger = logger_factory(role=Const.PASSIVE_NAME)

    # 1. Load dataset
    passive_inferset = NumpyDataset.from_mysql(
        role=Const.ACTIVE_NAME,
        dataset_type=Const.CLASSIFICATION,
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
        table=db_table_name,
        port=3306,
    )
    print(colored("1. Finish loading dataset.", "red"))
    logger.log("1. Finish loading dataset.")

    # 2. Feature transformation
    passive_inferset = scale(passive_inferset)
    print(colored("2. Finish transforming features", "red"))
    logger.log("2. Finish transforming features")

    # 3. Run PSI
    print(colored("3. PSI protocol started, computing...", "red"))
    messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )
    passive_psi = PassiveRSAPSI(
        messenger=messenger, logger=logger, num_workers=os.cpu_count()
    )
    common_ids = passive_psi.run(passive_inferset.ids)
    passive_inferset.filter(common_ids)
    print(colored("3. Finish psi protocol", "red"))
    logger.log("3. Finish psi protocol")

    # For online inference, you just need to substitute the model_name
    scores, preds = PassiveLogReg.online_inference(
        passive_inferset,
        model_name="20221205_160622-passive_party1-vertical_logreg-32561_samples.model",
        messenger=messenger,
    )
    print(scores)
    print(preds)

    # 6. Finish the whole pipeline
    messenger.close()
    print(colored("All Done.", "red"))
