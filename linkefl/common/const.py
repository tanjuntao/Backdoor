"""Store global constant variables"""


class Const:
    ACTIVE_NAME = 'active_party'
    PASSIVE_NAME = 'passive_party'

    RANDOM = 'random'
    SEQUENCE = 'sequence'
    IMPORTANCE = 'importance'

    START_SIGNAL = 'start'

    BUILDIN_DATASETS = ['cancer',
                       'digits',
                       'epsilon',
                       'census',
                       'credit',
                       'default_credit',
                       'covertype',
                       'higgs',
                       'diabetes',
                       'year',
                       'nyc-taxi',
                        'iris',
                        'wine',
                       'mnist',
                       'fashion_mnist',
                       'cifar',
                       'svhn']
    REGRESSION_DATASETS = ['diabetes']

    PLAIN = 'plain'
    PAILLIER = 'paillier'
    FAST_PAILLIER = 'fast_paillier'

    L1 = 'l1'
    L2 = 'l2'
    NONE = 'none'

    SOCKET = 'socket'
    FAST_SOCKET = 'fast_socket'

    VERTICAL_LINREG = 'vertical_linreg'
    VERTICAL_LOGREG = 'vertical_logreg'
    VERTICAL_SBT = 'vertical_sbt'
    VERTICAL_NN = 'vertical_nn'
    HORIZONTAL_NN = 'horizontal_nn'
    SPLIT_NN = 'split_nn'

    DEBUG = 'debug'
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'
    DATA_TYPE_DICT = {
        REGRESSION: ['diabetes', 'year', 'nyc-taxi'],
        CLASSIFICATION: ['cancer', 'digits', 'epsilon', 'census', 'credit',
                         'default_credit', 'covertype', 'higgs',
                         'iris', 'wine', 'mnist', 'fashion_mnist', 'cifar', 'svhn']
    }
