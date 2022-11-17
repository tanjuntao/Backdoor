"""Store global constant variables"""


class Const:
    ACTIVE_NAME = "active_party"
    PASSIVE_NAME = "passive_party"

    RANDOM = "random"
    SEQUENCE = "sequence"
    IMPORTANCE = "importance"

    START_SIGNAL = "start"
    PROJECT_CACHE_DIR = '.linkefl'

    BUILDIN_DATASETS = [
        "cancer",
        "digits",
        "epsilon",
        "census",
        "credit",
        "default_credit",
        "covertype",
        "higgs",
        "criteo",
        "diabetes",
        "year",
        "nyc_taxi",
        "iris",
        "wine",
        "mnist",
        "fashion_mnist",
        "cifar",
        "svhn",
        "avazu",
    ]
    REGRESSION_DATASETS = ["diabetes", "year", "nyc_taxi"]

    PLAIN = "plain"
    PAILLIER = "paillier"
    FAST_PAILLIER = "fast_paillier"
    RSA = 'rsa'

    L1 = "l1"
    L2 = "l2"
    NONE = "none"

    SOCKET = "socket"
    FAST_SOCKET = "fast_socket"
    FAST_SOCKET_V1 = 'fast_socket_v1'

    DATALOADER = "dataloader"
    TRANSFORM = "transform"
    RSA_PSI = "rsa_psi"
    CM20_PSI = "cm20_psi"
    VERTICAL_LINREG = "vertical_linreg"
    VERTICAL_LOGREG = "vertical_logreg"
    VERTICAL_SBT = "vertical_sbt"
    VERTICAL_NN = "vertical_nn"
    HORIZONTAL_NN = "horizontal_nn"
    SPLIT_NN = "split_nn"
    VERTICAL_INFERENCE = "vertical_inference"
    HORIZONTAL_INFERENCE = "horizontal_inference"

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    DATA_TYPE_DICT = {
        REGRESSION: ["diabetes", "year", "nyc_taxi"],
        CLASSIFICATION: [
            "cancer",
            "digits",
            "epsilon",
            "census",
            "credit",
            "criteo",
            "default_credit",
            "covertype",
            "higgs",
            "iris",
            "wine",
            "mnist",
            "fashion_mnist",
            "cifar",
            "svhn",
            "avazu",
        ],
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
