"""Global Configurations."""
import numpy as np


class LinearConfig:
    """Class to store all configurations."""

    ### Messenger configs ###
    ALICE_HOST = '127.0.0.1'
    ALICE_PORT = 22222
    BOB_HOST = '127.0.0.1'
    BOB_PORT = 22223
    MESSENGER_TYPE = 'fast_socket' # 'socket', 'fast_socket'
    SOCK_WAIT_INTERVAL = 1000 # used when messenger type is Socket

    ### Cryptosystem configs ###
    CRYPTO_TYPE = 'fast_paillier' # 'plain', 'paillier', 'fast_paillier'
    DEFAULT_KEY_SIZE = 1024
    NUM_ENC_ZEROS = 10000
    GEN_FROM_SET = False
    PRECISION = 0.01

    ### Training configs ###
    EPOCHS = 1
    BATCH_SIZE = 100
    LEARNING_RATE = 0.01
    PENALTY = 'l1' # 'none', 'l1', 'l2'
    LAMBDA = 0.001
    RANDOM_STATE = 0 # integer or None
    MULTI_THREADING = False

    ### Dataset configs ####
    ACC_DATASETS = ['breast_cancer', 'digits', 'epsilon', 'census_income']
    AUC_DATASETS = ['give_me_some_credit']
    # choose np_dataset name from
    # 'breast_cancer', 'digits', 'census_income', 'give_me_some_credit', 'epsilon'
    DATASET_NAME = 'give_me_some_credit'
    # the proportion of the number of features RSAPSIPassive has, which in range [0.0, 1.0]
    ATTACKER_FEATURES_FRAC = 0.5
    FEAT_SELECT_METHOD = 'sequential' # 'sequential', 'random', 'importance'

    if DATASET_NAME == 'breast_cancer':
        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = np.random.permutation(30)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = np.arange(30)
        elif FEAT_SELECT_METHOD == 'importance':
            pass

    if DATASET_NAME == 'census_income':
        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = np.random.permutation(81)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = np.arange(81)
        elif FEAT_SELECT_METHOD == 'importance':
            pass

    if DATASET_NAME == 'digits':
        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = np.random.permutation(64)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = np.arange(64)
        elif FEAT_SELECT_METHOD == 'importance':
            pass

    if DATASET_NAME == 'give_me_some_credit':
        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = np.random.permutation(10)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = np.arange(10)
        elif FEAT_SELECT_METHOD == 'importance':
            pass

    if DATASET_NAME == 'epsilon':
        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = np.random.permutation(100)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = np.arange(100)
        elif FEAT_SELECT_METHOD == 'importance':
            pass

    if DATASET_NAME == 'pseudo':
        PERMUTATION = np.arange(50)