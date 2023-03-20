import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import add_intercept, scale
from linkefl.psi.cm20 import ActiveCM20PSI
from linkefl.vfl.linear import ActiveLogReg

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_path = (  # substitute to your own abs csv path
        "/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/census-active-train.csv"
    )
    _has_header = False
    _test_size = 0.2
    _active_ips = [
        "localhost",
    ]
    _active_ports = [
        20000,
    ]
    _passive_ips = [
        "localhost",
    ]
    _passive_ports = [
        30000,
    ]
    _epochs = 10
    _batch_size = 32
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.001
    _crypto_type = Const.PLAIN
    _num_workers = 1
    _saving_model = True
    _random_state = 3047
    _key_size = 1024
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _messengers = [
        messenger_factory(
            messenger_type=Const.FAST_SOCKET,
            role=Const.ACTIVE_NAME,
            active_ip=ac_ip,
            active_port=ac_port,
            passive_ip=pass_ip,
            passive_port=pass_port,
        )
        for ac_ip, ac_port, pass_ip, pass_port in zip(
            _active_ips, _active_ports, _passive_ips, _passive_ports
        )
    ]
    _vfl_crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=100,
        gen_from_set=False,
    )
    task_start_time = time.time()

    # 1. Load dataset
    start_time = time.time()
    active_dataset = NumpyDataset.from_csv(
        role=Const.ACTIVE_NAME,
        abs_path=_dataset_path,
        dataset_type=Const.CLASSIFICATION,
        has_header=_has_header,
    )
    print(colored("1. Finish loading dataset.", "red"))
    _logger.log("1. Finish loading dataset.")
    _logger.log_component(
        name=Const.DATALOADER,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0,
    )

    # 2. Feature transformation
    start_time = time.time()
    active_dataset = scale(add_intercept(active_dataset))
    print(colored("2. Finish transforming features", "red"))
    _logger.log("2. Finish transforming features")
    _logger.log_component(
        name=Const.TRANSFORM,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0,
    )

    # 3. Run PSI
    print(colored("3. PSI protocol started, computing...", "red"))
    active_psi = ActiveCM20PSI(messengers=_messengers, logger=_logger)
    common_ids = active_psi.run(active_dataset.ids)
    print(f"length of common ids: {len(common_ids)}")
    active_dataset.filter(common_ids)
    active_trainset, active_testset = NumpyDataset.train_test_split(
        dataset=active_dataset, test_size=_test_size
    )
    print(colored("3. Finish psi protocol", "red"))
    _logger.log("3. Finish psi protocol")

    # 4. VFL training
    print(colored("4. Training protocol started, computing...", "red"))
    start_time = time.time()
    active_vfl = ActiveLogReg(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messengers=_messengers,
        cryptosystem=_vfl_crypto,
        logger=_logger,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        num_workers=_num_workers,
        saving_model=_saving_model,
    )
    active_vfl.train(active_trainset, active_testset)
    print(colored("4. Finish collaborative model training", "red"))
    _logger.log("4. Finish collaborative model training")
    _logger.log_component(
        name=Const.VERTICAL_LOGREG,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0,
    )

    # 5. VFL inference
    start_time = time.time()
    scores = active_vfl.predict(active_testset)
    print(
        "Acc: {:.5f} \nAuc: {:.5f} \nf1: {:.5f}".format(
            scores["acc"], scores["auc"], scores["f1"]
        )
    )
    print(colored("5. Finish collaborative inference", "red"))
    _logger.log(
        "Acc: {:.5f} Auc: {:.5f} f1: {:.5f}".format(
            scores["acc"], scores["auc"], scores["f1"]
        )
    )
    _logger.log("5. Finish collaborative inference")
    _logger.log_component(
        name=Const.VERTICAL_INFERENCE,
        status=Const.SUCCESS,
        begin=start_time,
        end=time.time(),
        duration=time.time() - start_time,
        progress=1.0,
    )

    # 6. Finish the whole pipeline
    for msger in _messengers:
        msger.close()
    print(colored("All Done.", "red"))
    _logger.log("All Done.")
    _logger.log_task(begin=task_start_time, end=time.time(), status=Const.SUCCESS)
