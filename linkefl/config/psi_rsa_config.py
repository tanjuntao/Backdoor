"""Global configurations"""
from pathlib import Path


class PSIRSAConfig:
    ### Messenger configs ###
    ALICE_HOST = '127.0.0.1'
    ALICE_PORT = 10001
    BOB_HOST = '127.0.0.1'
    BOB_PORT = 20001
    MESSENGER_TYPE = 'fast_socket' # choose from 'socket', 'fast_socket'
    SOCK_WAIT_INTERVAL = 1000 # only used when messenger type is 'socket'
    START_SIGNAL = 'start'

    ### RSA cryptosystem configs ###
    KEY_SIZE = 1024
    PUB_E = 0x10001
    # PUB_E = pow(2, 1000) - 1
    SECRET_CODE = 'zhaohang_ustc'
    PRIV_KEY_NAME = '.rsa_key.bin'

    ### Role names ###
    ROLE_ALICE = 'alice'
    ROLE_BOB = 'bob'

    ### ID set size ###
    ALICE_SIZE = 100_0000
    BOB_SIZE = 100