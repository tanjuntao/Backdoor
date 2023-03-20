import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale
from linkefl.messenger import FastSocket
from linkefl.psi.cm20 import PassiveCM20PSI
from linkefl.vfl.linear import PassiveLogReg

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_path = (  # substitute to your own abs csv path
        "/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/census-passive2-train.csv"
    )
    _has_header = False
    _test_size = 0.2
    _active_ip = "localhost"
    _active_port = 20001
    _passive_ip = "localhost"
    _passive_port = 30001
    _epochs = 10
    _batch_size = 32
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.001
    _rank = 2
    _num_workers = 1
    _saving_model = True
    _random_state = 3047
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=_active_ip,
        active_port=_active_port,
        passive_ip=_passive_ip,
        passive_port=_passive_port,
    )
    task_start_time = time.time()

    # 1. Load dataset
    start_time = time.time()
    passive_dataset = NumpyDataset.from_csv(
        role=Const.PASSIVE_NAME,
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
    passive_dataset = scale(passive_dataset)
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
    passive_psi = PassiveCM20PSI(messenger=_messenger, logger=_logger)
    common_ids = passive_psi.run(passive_dataset.ids)
    print(f"length of common ids: {len(common_ids)}")
    passive_dataset.filter(common_ids)
    passive_trainset, passive_testset = NumpyDataset.train_test_split(
        dataset=passive_dataset, test_size=_test_size
    )
    print(colored("3. Finish psi protocol", "red"))
    _logger.log("3. Finish psi protocol")

    # 4. VFL training
    print(colored("4. Training protocol started, computing...", "red"))
    start_time = time.time()
    passive_vfl = PassiveLogReg(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messenger=_messenger,
        logger=_logger,
        rank=_rank,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        num_workers=_num_workers,
        saving_model=_saving_model,
    )
    passive_vfl.train(passive_trainset, passive_testset)
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
    passive_vfl.predict(passive_testset)
    print(colored("5. Finish collaborative inference", "red"))
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
    _messenger.close()
    print(colored("All Done.", "red"))
    _logger.log("All Done.")
    _logger.log_task(begin=task_start_time, end=time.time(), status=Const.SUCCESS)
