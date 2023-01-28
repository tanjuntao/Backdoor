import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import RSAPSIPassive
from linkefl.vfl.tree import PassiveTreeParty

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
    task = "binary"
    crypto_type = Const.FAST_PAILLIER
    max_bin = 16
    colsample_bytree = 1
    logger = logger_factory(role=Const.PASSIVE_NAME)

    task_start_time = time.time()

    # 1. Load dataset
    start_time = time.time()
    passive_whole_dataset = NumpyDataset.from_mysql(
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
    logger.log_component(
        name=Const.DATALOADER,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0,
    )

    # 2. Feature transformation
    start_time = time.time()
    passive_whole_dataset = scale(passive_whole_dataset)
    print(colored("2. Finish transforming features", "red"))
    logger.log("2. Finish transforming features")
    logger.log_component(
        name=Const.TRANSFORM,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0,
    )

    # 3. Run PSI
    print(colored("3. PSI protocol started, computing...", "red"))
    messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )
    passive_psi = RSAPSIPassive(messenger, logger)
    common_ids = passive_psi.run(passive_whole_dataset.ids)
    passive_whole_dataset.filter(common_ids)
    passive_trainset, passive_testset = NumpyDataset.train_test_split(
        dataset=passive_whole_dataset, test_size=0.2
    )
    print(colored("3. Finish psi protocol", "red"))
    logger.log("3. Finish psi protocol")

    # 4. VFL training
    print(colored("4. Training protocol started, computing...", "red"))
    start_time = time.time()
    passive_vfl = PassiveTreeParty(
        task=task,
        crypto_type=crypto_type,
        messenger=messenger,
        logger=logger,
        max_bin=max_bin,
        colsample_bytree=colsample_bytree,
        saving_model=True,
        model_name="passive1_tree_model.model",
    )
    passive_vfl.train(passive_trainset, passive_testset)
    print(colored("4. Finish collaborative model training", "red"))
    logger.log("4. Finish collaborative model training")
    logger.log_component(
        name=Const.VERTICAL_LOGREG,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0,
    )

    # VFL inference
    start_time = time.time()
    passive_vfl.predict(passive_testset)
    print(colored("5. Finish collaborative inference", "red"))
    logger.log("5. Finish collaborative inference")
    logger.log_component(
        name=Const.VERTICAL_INFERENCE,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0,
    )

    # 6. Finish the whole pipeline
    messenger.close()
    print(colored("All Done.", "red"))
    logger.log("All Done.")
    logger.log_task(begin=task_start_time, end=time.time(), status=Const.SUCCESS)
