import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
from linkefl.crypto import RSA
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import add_intercept, scale
from linkefl.psi.rsa import RSAPSIActive
from linkefl.vfl.tree import ActiveTreeParty

if __name__ == "__main__":
    # 0. Set parameters
    db_host = "localhost"
    db_user = "tiger"
    db_name = "hello_db"
    db_table_name = "hello_table"
    db_password = "hello_pw"
    active_ip = ["localhost", "localhost"]
    active_port = [20000, 30000]
    passive_ip = ["localhost", "localhost"]
    passive_port = [20001, 30001]
    n_trees = 1
    task = "binary"
    n_labels = 2
    crypto_type = Const.FAST_PAILLIER
    learning_rate = 0.3
    compress = False
    max_bin = 16
    max_depth = 4
    reg_lambda = 0.1
    min_split_samples = 3
    min_split_gain = 1e-7
    fix_point_precision = 53
    sampling_method = "uniform"
    subsample = 1
    top_rate = 0.5
    other_rate = 0.5
    colsample_bytree = 1
    n_processes = 6
    key_size = 1024
    logger = logger_factory(role=Const.ACTIVE_NAME)

    task_start_time = time.time()

    # 1. Load dataset
    start_time = time.time()
    active_whole_dataset = NumpyDataset.from_mysql(
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
    active_whole_dataset = scale(add_intercept(active_whole_dataset))
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
    messengers = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=ac_ip,
            active_port=ac_port,
            passive_ip=pass_ip,
            passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in zip(
            active_ip, active_port, passive_ip, passive_port
        )
    ]
    psi_crypto = RSA()
    active_psi = RSAPSIActive(messengers, psi_crypto, logger)
    common_ids = active_psi.run(active_whole_dataset.ids)
    active_whole_dataset.filter(common_ids)
    active_trainset, active_testset = NumpyDataset.train_test_split(
        dataset=active_whole_dataset, test_size=0.2
    )
    print(colored("3. Finish psi protocol", "red"))
    logger.log("3. Finish psi protocol")

    # 4. VFL training
    print(colored("4. Training protocol started, computing...", "red"))
    start_time = time.time()
    vfl_crypto = crypto_factory(
        crypto_type=crypto_type,
        key_size=key_size,
        num_enc_zeros=10000,
        gen_from_set=False,
    )
    active_vfl = ActiveTreeParty(
        n_trees=n_trees,
        task=task,
        n_labels=n_labels,
        crypto_type=crypto_type,
        crypto_system=vfl_crypto,
        messengers=messengers,
        logger=logger,
        learning_rate=learning_rate,
        compress=compress,
        max_bin=max_bin,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        min_split_samples=min_split_samples,
        min_split_gain=min_split_gain,
        fix_point_precision=fix_point_precision,
        sampling_method=sampling_method,
        subsample=subsample,
        top_rate=top_rate,
        other_rate=other_rate,
        colsample_bytree=colsample_bytree,
        n_processes=n_processes,
        saving_model=True,
        model_name="active_tree_model.model",
    )
    active_vfl.train(active_trainset, active_testset)
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

    # 5. VFL inference
    start_time = time.time()
    scores = active_vfl.score(active_testset)
    print(
        "Acc: {:.5f} \nAuc: {:.5f} \nf1: {:.5f}".format(
            scores["acc"], scores["auc"], scores["f1"]
        )
    )
    print(colored("5. Finish collaborative inference", "red"))
    logger.log(
        "Acc: {:.5f} Auc: {:.5f} f1: {:.5f}".format(
            scores["acc"], scores["auc"], scores["f1"]
        )
    )
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
    for messenger in messengers:
        messenger.close()
    print(colored("All Done.", "red"))
    logger.log("All Done.")
    logger.log_task(begin=task_start_time, end=time.time(), status=Const.SUCCESS)
