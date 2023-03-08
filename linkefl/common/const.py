"""Store global constant variables"""


class Const:
    ACTIVE_NAME = "active_party"
    PASSIVE_NAME = "passive_party"

    RANDOM = "random"
    SEQUENCE = "sequence"
    IMPORTANCE = "importance"

    START_SIGNAL = "start"
    PROJECT_CACHE_DIR = ".linkefl"

    BUILDIN_DATASETS = [
        "avazu",
        "cancer",
        "census",
        "cifar10",
        "covertype",
        "credit",
        "criteo",
        "default_credit",
        "diabetes",
        "digits",
        "epsilon",
        "fashion_mnist",
        "higgs",
        "iris",
        "mnist",
        "nyc_taxi",
        "svhn",
        "tab_fashion_mnist",
        "tab_mnist",
        "wine",
        "year",
    ]

    PLAIN = "plain"
    PAILLIER = "paillier"
    FAST_PAILLIER = "fast_paillier"
    RSA = "rsa"

    L1 = "l1"
    L2 = "l2"
    NONE = "none"

    SOCKET = "socket"
    FAST_SOCKET = "fast_socket"
    FAST_SOCKET_V1 = "fast_socket_v1"

    DATALOADER = "dataloader"
    TRANSFORM = "transform"
    RSA_PSI = "rsa_psi"
    CM20_PSI = "cm20_psi"
    VERTICAL_LINREG = "vertical_linreg"
    VERTICAL_LOGREG = "vertical_logreg"
    VERTICAL_SBT = "vertical_sbt"
    VERTICAL_LIGHTGBM = "vertical_lightgbm"
    VERTICAL_NN = "vertical_nn"
    HORIZONTAL_NN = "horizontal_nn"
    SPLIT_NN = "split_nn"
    VERTICAL_INFERENCE = "vertical_inference"
    HORIZONTAL_INFERENCE = "horizontal_inference"

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    PYTORCH_DATASET = [
        "cifar10",
        "fashion_mnist",
        "mnist",
        "svhn",
        "tab_mnist",
        "tab_fashion_mnist",
    ]
    REGRESSION_DATASETS = [
        "diabetes",
        "nyc_taxi",
        "year",
    ]
    DATA_TYPE_DICT = {
        REGRESSION: [
            "diabetes",
            "nyc_taxi",
            "year",
        ],
        CLASSIFICATION: list(set(BUILDIN_DATASETS) - set(REGRESSION_DATASETS)),
    }

    BLOSC = "blosc"
    ZLIB = "zlib"
    COMPRESSION_DICT = {
        BLOSC: 0,
        ZLIB: 1,
    }

    RUNNING = "running"
    FAILURE = "failure"
    SUCCESS = "success"
