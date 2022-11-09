from termcolor import colored

from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, logger_factory
from linkefl.crypto import RSACrypto
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import parse_label
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import RSAPSIActive
from linkefl.vfl.tree import ActiveTreeParty

if __name__ == "__main__":
    # 0. Set parameters
    trainset_path = r"../../linkefl/data/tabular/give_me_some_credit_active_train.csv"
    testset_path = r"../../linkefl/data/tabular/give_me_some_credit_active_test.csv"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE

    active_ip = "localhost"
    active_port = 20000
    passive_ip = "localhost"
    passive_port = 30000

    _n_trees = 5
    _task = "binary"
    _n_labels = 2
    _crypto_type = Const.PAILLIER
    _learning_rate = 0.3
    _compress = False
    _max_bin = 16
    _max_depth = 4
    _reg_lambda = 0.1
    _min_split_samples = 3
    _min_split_gain = 1e-7
    _fix_point_precision = 53
    _sampling_method = "uniform"
    _n_processes = 6

    _key_size = 1024
    _logger = logger_factory(role=Const.ACTIVE_NAME)

    # 1. Load dataset
    active_trainset = NumpyDataset.from_csv(role=Const.ACTIVE_NAME,
                                            abs_path=trainset_path,
                                            dataset_type=Const.CLASSIFICATION)
    active_testset = NumpyDataset.from_csv(role=Const.ACTIVE_NAME,
                                           abs_path=testset_path,
                                           dataset_type=Const.CLASSIFICATION)
    print(colored("1. Finish loading dataset.", "red"))

    # 2. Feature transformation
    active_trainset = parse_label(active_trainset)
    active_testset = parse_label(active_testset)
    print(colored("2. Finish transforming features", "red"))

    # 3. Run PSI
    messenger = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )
    psi_crypto = RSACrypto()
    active_psi = RSAPSIActive([messenger],
                              psi_crypto,
                              _logger,
                              num_workers=_n_processes)
    common_ids = active_psi.run(active_trainset.ids)
    active_trainset.filter(common_ids)
    print(colored("3. Finish psi protocol", "red"))

    # 4. VFL training
    vfl_crypto = crypto_factory(crypto_type=_crypto_type, key_size=_key_size, num_enc_zeros=10000, gen_from_set=False)
    active_vfl = ActiveTreeParty(
        n_trees=_n_trees,
        task=_task,
        n_labels=_n_labels,
        crypto_type=_crypto_type,
        crypto_system=vfl_crypto,
        messengers=[messenger],
        learning_rate=_learning_rate,
        compress=_compress,
        max_bin=_max_bin,
        max_depth=_max_depth,
        reg_lambda=_reg_lambda,
        min_split_samples=_min_split_samples,
        min_split_gain=_min_split_gain,
        fix_point_precision=_fix_point_precision,
        sampling_method=_sampling_method,
        n_processes=_n_processes,
    )
    active_vfl.train(active_trainset, active_testset)
    print(colored("4. Finish collaborative model training", "red"))

    # 5. VFL inference
    scores = active_vfl.predict(active_testset)
    print(scores)
    # print("Acc: {:.5f} \nAuc: {:.5f} \nf1: {:.5f}".format(scores["acc"], scores["auc"], scores["f1"]))
    print(colored("5. Finish collaborative inference", "red"))

    # 6. Finish the whole pipeline
    messenger.close()
    print(colored("All Done.", "red"))
