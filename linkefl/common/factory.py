from linkefl.config import LinearConfig
from linkefl.crypto import Plain, Paillier, FastPaillier
from linkefl.crypto import PartialPlain, PartialPaillier, PartialFastPaillier
from linkefl.messenger import SocketMessenger, FastSocketMessenger


def crypto_factory():
    if LinearConfig.CRYPTO_TYPE == 'plain':
        crypto = Plain(key_size=None)
    elif LinearConfig.CRYPTO_TYPE == 'paillier':
        crypto = Paillier(LinearConfig, key_size=LinearConfig.DEFAULT_KEY_SIZE)
    elif LinearConfig.CRYPTO_TYPE == 'fast_paillier':
        crypto = FastPaillier(LinearConfig,
                              key_size=LinearConfig.DEFAULT_KEY_SIZE,
                              num_enc_zeros=LinearConfig.NUM_ENC_ZEROS,
                              gen_from_set=LinearConfig.GEN_FROM_SET)
    else:
        raise ValueError('Unrecoginized crypto type.')

    return crypto


def partial_crypto_factory(public_key):
    if LinearConfig.CRYPTO_TYPE == 'plain':
        partial_crypto = PartialPlain(pub_key=public_key)
    elif LinearConfig.CRYPTO_TYPE == 'paillier':
        partial_crypto = PartialPaillier(pub_key=public_key)
    elif LinearConfig.CRYPTO_TYPE == 'fast_paillier':
        partial_crypto = PartialFastPaillier(pub_key=public_key,
                                             num_enc_zeros=LinearConfig.NUM_ENC_ZEROS,
                                             gen_from_set=LinearConfig.GEN_FROM_SET)
    else:
        raise ValueError('Unrecoginized crypto type.')

    return partial_crypto


def messenger_factory(role):
    if LinearConfig.MESSENGER_TYPE == 'socket':
        messenger = SocketMessenger(role=role)
    elif LinearConfig.MESSENGER_TYPE == 'fast_socket':
        messenger = FastSocketMessenger(role=role)
    else:
        raise ValueError('Unrecoginized messenger type.')

    return messenger