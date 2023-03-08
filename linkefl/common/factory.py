from typing import Optional

from linkefl.common.const import Const
from linkefl.common.log import GlobalLogger
from linkefl.crypto import (
    FastPaillier,
    Paillier,
    PartialFastPaillier,
    PartialPaillier,
    PartialPlain,
    Plain,
)
from linkefl.messenger.socket import FastSocket, FastSocket_v1, Socket
from linkefl.messenger.socket_disconnection import (
    FastSocket_disconnection,
    FastSocket_disconnection_v1,
    Socket_disconnection,
)
from linkefl.messenger.socket_multiclient import (
    FastSocket_multi_disconnection_v1,
    FastSocket_multi_v1,
)


def crypto_factory(
    crypto_type, *, key_size=1024, num_enc_zeros=10000, gen_from_set=True
):
    if crypto_type == Const.PLAIN:
        crypto = Plain(key_size=key_size)
    elif crypto_type == Const.PAILLIER:
        crypto = Paillier(key_size=key_size)
    elif crypto_type == Const.FAST_PAILLIER:
        crypto = FastPaillier(
            key_size=key_size, num_enc_zeros=num_enc_zeros, gen_from_set=gen_from_set
        )
    else:
        raise ValueError("Unrecoginized crypto type.")

    return crypto


def partial_crypto_factory(
    crypto_type, *, public_key, num_enc_zeros=10000, gen_from_set=True
):
    if crypto_type == Const.PLAIN:
        partial_crypto = PartialPlain(raw_public_key=public_key)
    elif crypto_type == Const.PAILLIER:
        partial_crypto = PartialPaillier(raw_public_key=public_key)
    elif crypto_type == Const.FAST_PAILLIER:
        partial_crypto = PartialFastPaillier(
            raw_public_key=public_key,
            num_enc_zeros=num_enc_zeros,
            gen_from_set=gen_from_set,
        )
    else:
        raise ValueError("Unrecoginized crypto type.")

    return partial_crypto


def messenger_factory(
    messenger_type,
    *,
    role,
    active_ip,
    active_port,
    passive_ip,
    passive_port,
    verbose=False,
):
    if messenger_type == Const.SOCKET:
        messenger = Socket(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    else:
        raise ValueError("Unrecoginized messenger type.")

    return messenger


def messenger_factory_disconnection(
    messenger_type,
    *,
    role,
    active_ip,
    active_port,
    passive_ip,
    passive_port,
    verbose=False,
    model_type="Tree",
):
    if messenger_type == Const.SOCKET:
        messenger = Socket_disconnection(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket_disconnection(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    elif messenger_type == Const.FAST_SOCKET_V1:
        messenger = FastSocket_disconnection_v1(
            role=role,
            model_type=model_type,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    else:
        raise ValueError("Unrecoginized messenger type.")

    return messenger


def messenger_factory_v1(
    messenger_type,
    *,
    role,
    active_ip,
    active_port,
    passive_ip,
    passive_port,
    verbose=False,
):
    if messenger_type == Const.SOCKET:
        messenger = Socket(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket_v1(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    else:
        raise ValueError("Unrecoginized messenger type.")

    return messenger


def messenger_factory_multi(
    messenger_type,
    *,
    role,
    active_ip,
    active_port,
    passive_ip,
    passive_port,
    verbose=False,
    world_size=1,
    rank=1,
):
    if messenger_type == Const.SOCKET:
        messenger = Socket(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket_multi_v1(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            world_size=world_size,
            verbose=verbose,
        )
    else:
        raise ValueError("Unrecoginized messenger type.")

    return messenger


def messenger_factory_multi_disconnection(
    messenger_type,
    *,
    role,
    model_type,
    active_ip,
    active_port,
    passive_ip,
    passive_port,
    verbose=False,
    world_size=1,
    rank=1,
):
    if messenger_type == Const.SOCKET:
        messenger = Socket(
            role=role,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            verbose=verbose,
        )
    elif messenger_type == Const.FAST_SOCKET:
        messenger = FastSocket_multi_disconnection_v1(
            role=role,
            model_type=model_type,
            active_ip=active_ip,
            active_port=active_port,
            passive_ip=passive_ip,
            passive_port=passive_port,
            world_size=world_size,
            verbose=verbose,
        )
    else:
        raise ValueError("Unrecoginized messenger type.")

    return messenger


def logger_factory(
    role: str,
    writing_file: bool = False,
    file_path: Optional[str] = None,
    remote_url: Optional[str] = None,
    stacklevel: int = 2,
):
    return GlobalLogger(
        role=role,
        writing_file=writing_file,
        file_path=file_path,
        remote_url=remote_url,
        stacklevel=stacklevel
    )
