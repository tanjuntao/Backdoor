from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.crypto import Plain, Paillier, FastPaillier
from linkefl.crypto import PartialPlain, PartialPaillier, PartialFastPaillier
from linkefl.messenger import Socket, FastSocket

from linkefl.messenger.socket import Socket, FastSocket,FastSocket_v1
from linkefl.messenger.socket_disconnection import Socket_disconnection, FastSocket_disconnection
from linkefl.messenger.socket_multiclient import FastSocket_multi_v1

def crypto_factory(crypto_type,
                   *,
                   key_size=1024,
                   num_enc_zeros=10000,
                   gen_from_set=True):
    if crypto_type == Const.PLAIN:
        crypto = Plain(key_size=key_size)
    elif crypto_type == Const.PAILLIER:
        crypto = Paillier(key_size=key_size)
    elif crypto_type == Const.FAST_PAILLIER:
        crypto = FastPaillier(key_size=key_size,
                              num_enc_zeros=num_enc_zeros,
                              gen_from_set=gen_from_set)
    else:
        raise ValueError('Unrecoginized crypto type.')

    return crypto


def partial_crypto_factory(crypto_type,
                           *,
                           public_key,
                           num_enc_zeros=10000,
                           gen_from_set=True):
    if crypto_type == Const.PLAIN:
        partial_crypto = PartialPlain(raw_public_key=public_key)
    elif crypto_type == Const.PAILLIER:
        partial_crypto = PartialPaillier(raw_public_key=public_key)
    elif crypto_type == Const.FAST_PAILLIER:
        partial_crypto = PartialFastPaillier(raw_public_key=public_key,
                                             num_enc_zeros=num_enc_zeros,
                                             gen_from_set=gen_from_set)
    else:
        raise ValueError('Unrecoginized crypto type.')

    return partial_crypto


def messenger_factory(messenger_type,
                      *,
                      role,
                      active_ip,
                      active_port,
                      passive_ip,
                      passive_port,
                      verbose=False):
    if messenger_type == Const.SOCKET:
        messenger = Socket(role=role,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port,
                           verbose=verbose)
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket(role=role,
                               active_ip=active_ip,
                               active_port=active_port,
                               passive_ip=passive_ip,
                               passive_port=passive_port,
                               verbose=verbose)
    else:
        raise ValueError('Unrecoginized messenger type.')

    return messenger


def messenger_factory_disconnection(messenger_type,
                      *,
                      role,
                      active_ip,
                      active_port,
                      passive_ip,
                      passive_port,
                      verbose=False):
    if messenger_type == Const.SOCKET:
        messenger = Socket_disconnection(role=role,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port,
                           verbose=verbose)
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket_disconnection(role=role,
                               active_ip=active_ip,
                               active_port=active_port,
                               passive_ip=passive_ip,
                               passive_port=passive_port,
                               verbose=verbose)
    else:
        raise ValueError('Unrecoginized messenger type.')

    return messenger

def messenger_factory_v1(messenger_type,
                      *,
                      role,
                      active_ip,
                      active_port,
                      passive_ip,
                      passive_port,
                      verbose=False):
    if messenger_type == Const.SOCKET:
        messenger = Socket(role=role,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port,
                           verbose=verbose)
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket_v1(role=role,
                               active_ip=active_ip,
                               active_port=active_port,
                               passive_ip=passive_ip,
                               passive_port=passive_port,
                               verbose=verbose)
    else:
        raise ValueError('Unrecoginized messenger type.')

    return messenger

def messenger_factory_multi(messenger_type,
                      *,
                      role,
                      active_ip,
                      active_port,
                      passive_ip,
                      passive_port,
                      verbose=False,
                      world_size=1,
                      rank=1):
    if messenger_type == Const.SOCKET:
        messenger = Socket(role=role,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port,
                           verbose=verbose)
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket_multi_v1(role=role,
                               active_ip=active_ip,
                               active_port=active_port,
                               passive_ip=passive_ip,
                               passive_port=passive_port,
                               world_size=world_size,
                               verbose=verbose,)
    else:
        raise ValueError('Unrecoginized messenger type.')

    return messenger


def logger_factory(role,
                   writing_file=False,
                   writing_http=False,
                   http_host=None, http_port=None, http_url=None):

    return GlobalLogger(role=role,
                        writing_file=writing_file,
                        writing_http=writing_http,
                        http_host=http_host,
                        http_port=http_port,
                        http_url=http_url)
