import os
import time

from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale
from linkefl.messenger import FastSocket
from linkefl.psi import PassiveCM20PSI
from linkefl.vfl.tree import PassiveTreeParty

if __name__ == "__main__":
    # 0. Set parameters
    _dataset_path = (  # substitute to your own abs csv path
        "/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/census-passive2-train.csv"
    )
    _has_header = False
    _test_size = 0.2
    active_ip = "localhost"
    active_port = 20001
    passive_ip = "localhost"
    passive_port = 30001
    task = "binary"
    crypto_type = Const.FAST_PAILLIER
    max_bin = 16
    colsample_bytree = 1
    saving_model = True
    n_processes = os.cpu_count()
    _logger = logger_factory(role=Const.PASSIVE_NAME)
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
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
    passive_vfl = PassiveTreeParty(
        task=task,
        crypto_type=crypto_type,
        messenger=_messenger,
        logger=_logger,
        max_bin=max_bin,
        colsample_bytree=colsample_bytree,
        n_processes=n_processes,
        saving_model=saving_model,
        model_name="passive2_tree_model.model",
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
