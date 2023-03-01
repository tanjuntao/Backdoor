from linkefl.common.const import Const
from linkefl.common.factory import crypto_factory, messenger_factory
from linkefl.dataio import NumpyDataset
from linkefl.feature.transform import scale
from linkefl.feature.woe import PassiveWoe
from linkefl.feature.chi_square_bin import PassiveChiBin
from linkefl.feature.pearson import PassivePearsonVfl
from linkefl.dataio import TorchDataset


if __name__ == "__main__":
    # Active
    # 0. Set parameters
    #  binary: cancer, digits, epsilon, census, credit, default_credit, criteo
    #  regression: diabetes
    dataset_name = "census"
    passive_feat_frac = 0.1
    feat_perm_option = Const.SEQUENCE
    crypto_type = Const.FAST_PAILLIER
    key_size = 1024
    active_ip = 'localhost'
    active_port = 20000
    passive_ip = 'localhost'
    passive_port = 20002

    # Passive
    # 1. Loading datasets and preprocessing
    print('Loading dataset...')
    # passive_trainset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
    #                                                 dataset_name=dataset_name,
    #                                                 root='../vfl/data',
    #                                                 train=True,
    #                                                 download=True,
    #                                                 passive_feat_frac=passive_feat_frac,
    #                                                 feat_perm_option=feat_perm_option)
    # passive_testset = NumpyDataset.buildin_dataset(role=Const.PASSIVE_NAME,
    #                                                dataset_name=dataset_name,
    #                                                root='../vfl/data',
    #                                                train=False,
    #                                                download=True,
    #                                                passive_feat_frac=passive_feat_frac,
    #                                                feat_perm_option=feat_perm_option)
    passive_trainset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.PASSIVE_NAME,
        root="../vfl/data",
        train=True,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    passive_testset = TorchDataset.buildin_dataset(
        dataset_name=dataset_name,
        role=Const.PASSIVE_NAME,
        root="../vfl/data",
        train=False,
        download=True,
        passive_feat_frac=passive_feat_frac,
        feat_perm_option=feat_perm_option,
    )
    # passive_trainset = NumpyDataset.feature_split(passive_trainset, n_splits=2)[0]
    # passive_testset = NumpyDataset.feature_split(passive_testset, n_splits=2)[0]
    # passive_trainset = scale(passive_trainset)
    # passive_testset = scale(passive_testset)
    print("Done")

    # 2. Initialize messenger
    _crypto = crypto_factory(crypto_type=crypto_type,
                             key_size=key_size,
                             num_enc_zeros=10,
                             gen_from_set=False)

    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.PASSIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)
    # print(passive_trainset.n_features)
    # 明文计算被动方本地woe & iv值
    # split, woe, iv = PassiveWoe(dataset=passive_trainset, idxes=[2,3], messenger=_messenger).cal_woe()
    # print(split, woe, iv)
    # 密文计算被动方woe & iv值
    PassiveWoe(dataset=passive_trainset, idxes=[2, 3], messenger=_messenger, cryptosystem=_crypto).cal_woe_encry()

    # chi_bin = PassiveChiBin(dataset=passive_trainset, idxes=[2,3], messenger=_messenger, max_group=200).chi_bin()
    # print(chi_bin)

    # pearson = PassivePearsonVfl(passive_trainset, _messenger).pearson_vfl()
    # print(pearson)