from linkefl.common.const import Const
from linkefl.common.factory import (
    crypto_factory,
    logger_factory,
    messenger_factory,
    messenger_factory_multi_disconnection,
)
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import add_intercept, parse_label, scale
from linkefl.vfl.linear.logreg.multi_disconnection import ActiveLogReg_disconnection

if __name__ == "__main__":
    # 0. Set parameters
    dataset_name = "census"
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    active_ip = ["localhost", "localhost"]
    active_port = [20000, 30000]
    passive_ip = ["localhost", "localhost"]
    passive_port = [20001, 30001]
    reconnection_port = [20002, 30002]
    world_size = 2
    _epochs = 100
    _batch_size = 100
    _learning_rate = 0.01
    _penalty = Const.L2
    _reg_lambda = 0.01
    _crypto_type = Const.PLAIN
    _random_state = 3347
    _key_size = 1024
    _using_pool = False
    reconnection = True
    saving_model = True

    # 1. Loading datasets and preprocessing
    # Option 1: Scikit-Learn style
    print("Loading dataset...")
    active_trainset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root="../../data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    active_testset = NumpyDataset.buildin_dataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root="../../data",
        train=False,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    # load dummy dataset
    # dummy_dataset = NumpyDataset.dummy_daaset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_type=Const.CLASSIFICATION,
    #     n_samples=100000,
    #     n_features=100,
    #     passive_feat_frac=passive_feat_frac
    # )
    # active_trainset, active_testset = NumpyDataset.train_test_split(
    #     dummy_dataset,
    #     test_size=0.2
    # )

    # if using credit dataset, remember to apply scale after add_intercept,
    # otherwise the model cannot converge
    active_trainset = add_intercept(scale(parse_label(active_trainset)))
    active_testset = add_intercept(scale(parse_label(active_testset)))
    print("Done.")
    # Option 2: PyTorch style
    # print('Loading dataset...')
    # transform = Compose([ParseLabel(), Scale(), AddIntercept()])
    # active_trainset = NumpyDataset.buildin_dataset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_name=dataset_name,
    #     train=True,
    #     passive_feat_frac=passive_feat_frac,
    #     feat_perm_option=feat_perm_option,
    #     trainsform=transform
    # )
    # active_testset = NumpyDataset.buildin_dataset(
    #     role=Const.ACTIVE_NAME,
    #     dataset_name=dataset_name,
    #     train=False,
    #     passive_feat_frac=passive_feat_frac,
    #     feat_perm_option=feat_perm_option,
    #     transform=transform
    # )
    # print('Done.')

    # 3. Initialize cryptosystem
    _crypto = crypto_factory(
        crypto_type=_crypto_type,
        key_size=_key_size,
        num_enc_zeros=10,
        gen_from_set=False,
    )

    # 4. Initialize messenger
    # _messenger = [
    #     messenger_factory(messenger_type=Const.FAST_SOCKET,
    #                       role=Const.ACTIVE_NAME,
    #                       active_ip=ac_ip,
    #                       active_port=ac_port,
    #                       passive_ip=pass_ip,
    #                       passive_port=pass_port,
    #     )
    #     for ac_ip, ac_port, pass_ip, pass_port in
    #         zip(active_ip, active_port, passive_ip, passive_port)
    # ]
    _messenger = messenger_factory_multi_disconnection(
        messenger_type=Const.FAST_SOCKET,
        role=Const.ACTIVE_NAME,
        model_type="NN",
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
        world_size=world_size,
    )

    print("ACTIVE PARTY started, connecting...")

    # 5. Initialize model and start training
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    active_party = ActiveLogReg_disconnection(
        epochs=_epochs,
        batch_size=_batch_size,
        learning_rate=_learning_rate,
        messenger=_messenger,
        cryptosystem=_crypto,
        logger=_logger,
        penalty=_penalty,
        reg_lambda=_reg_lambda,
        random_state=_random_state,
        using_pool=_using_pool,
        saving_model=saving_model,
        world_size=world_size,
        reconnection=reconnection,
        reconnection_port=reconnection_port,
    )

    active_party.train(active_trainset, active_testset)

    # 6. Close messenger, finish training.
    # for msger_ in _messenger:
    #     msger_.close()
    _messenger.close()

    # For online inference, you just need to substitute the model_name
    # scores = ActiveLogReg.online_inference(
    #     active_testset,
    #     model_name='20220831_185054-active_party-vertical_logreg-455_samples.model',
    #     messenger=_messenger
    # )
    #
    # print(scores)
