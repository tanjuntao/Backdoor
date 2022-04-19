import torch


class NNConfig:
    ALICE_HOST = '127.0.0.1'
    ALICE_PORT = 22222
    ALICE_DEVICE = 'cpu'
    BOB_HOST = '127.0.0.1'
    BOB_PORT = 22223
    BOB_DEVICE = 'cpu'

    BATCH_SIZE = 64
    EPOCHS = 80
    LEARNING_RATE = 1e-2
    TRAINING_VERBOSE = True

    # choose from ['census_income', 'give_me_some_credit', 'breast_cancer'
    # 'digits', 'fashion_mnist', 'mnist']
    DATASET_NAME = 'mnist'

    DETECT_THRESH_RATIO = 0.2
    SOCK_WAIT_INTERVAL = 800
    Q = 2000
    ATTACKER_FEATURES_FRAC = 0.5
    FEAT_SELECT_METHOD = 'random'
    if FEAT_SELECT_METHOD == 'random':
        SEED = 0
        torch.manual_seed(SEED) # reproducibility


    if DATASET_NAME in ('mnist', 'fashion_mnist'):
        __num_alice_input = int(28*28*ATTACKER_FEATURES_FRAC)
        __num_bob_input = 28*28 - __num_alice_input
        ALICE_BOTTOM_NODES = [__num_alice_input, 256, 128]
        BOB_BOTTOM_NODES = [__num_bob_input, 256, 128]
        INTERSECTION_NODES = [128, 128, 10]
        TOP_NODES = [10, 10]

        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = torch.randperm(28*28)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = torch.arange(28*28)
        elif FEAT_SELECT_METHOD == 'importance':
            # PERMUTATION = feature_ranking(DATASET_NAME, measurement='xgboost')
            pass
        else:
            pass

    elif DATASET_NAME == 'census_income':
        __num_alice_input = int(81 * ATTACKER_FEATURES_FRAC)
        __num_bob_input = 81 - __num_alice_input
        ALICE_BOTTOM_NODES = [__num_alice_input, 20, 10]
        BOB_BOTTOM_NODES = [__num_bob_input, 20, 10]
        INTERSECTION_NODES = [10, 10, 10]
        TOP_NODES = [10, 2]

        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = torch.randperm(81)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = torch.arange(81)
        elif FEAT_SELECT_METHOD == 'importance':
            # PERMUTATION = feature_ranking(DATASET_NAME, measurement='xgboost')
            pass
        else:
            pass

    elif DATASET_NAME == 'give_me_some_credit':
        __num_alice_input = int(10 * ATTACKER_FEATURES_FRAC)
        __num_bob_input = 10 - __num_alice_input
        ALICE_BOTTOM_NODES = [__num_alice_input, 3]
        BOB_BOTTOM_NODES = [__num_bob_input, 3]
        INTERSECTION_NODES = [3, 3, 6]
        TOP_NODES = [6, 3, 2]

        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = torch.randperm(10)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = torch.arange(10)
        elif FEAT_SELECT_METHOD == 'importance':
            # PERMUTATION = feature_ranking(DATASET_NAME, measurement='xgboost')
            pass
        else:
            pass

    elif DATASET_NAME == 'digits':
        __num_alice_input = int(64 * ATTACKER_FEATURES_FRAC)
        __num_bob_input = 64 - __num_alice_input
        ALICE_BOTTOM_NODES = [__num_alice_input, 20, 16]
        BOB_BOTTOM_NODES = [__num_bob_input, 20, 16]
        INTERSECTION_NODES = [16, 16, 16]
        TOP_NODES = [16, 10]

        if FEAT_SELECT_METHOD == 'random':
            PERMUTATION = torch.randperm(64)
        elif FEAT_SELECT_METHOD == 'sequential':
            PERMUTATION = torch.arange(64)
        elif FEAT_SELECT_METHOD == 'importance':
            # PERMUTATION = feature_ranking(DATASET_NAME, measurement='xgboost')
            pass
        else:
            pass

    else:
        pass

    # census_income: [10, 8, 5, 2]
    # mnist: [128, 100, 64, 10]
    APPEND_NODES = [128, 100, 64, 10]
    # APPEND_NODES = [10, 5, 2]
    # APPEND_NODES = [3, 2]
    # APPEND_NODES = [10, 2]
    LOCAL_EPOCHS = 200
    LOCAL_LEARNING_RATE = 5 * 1e-2

    ### Cryptosystem configs ###
    CRYPTO_TYPE = 'plain'  # 'plain', 'paillier', 'fast_paillier'
    DEFAULT_KEY_SIZE = 1024
    NUM_ENC_ZEROS = 10000
    GEN_FROM_SET = False