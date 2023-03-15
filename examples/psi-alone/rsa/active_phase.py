import argparse
import os

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import RSA
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi import ActiveRSAPSI

if __name__ == "__main__":
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str)
    args = parser.parse_args()

    active_ips = [
        "127.0.0.1",
    ]
    active_ports = [
        20000,
    ]
    passive_ips = [
        "127.0.0.1",
    ]
    passive_ports = [
        30000,
    ]
    _messengers = [
        FastSocket(
            role=Const.ACTIVE_NAME,
            active_ip=active_ip_,
            active_port=active_port_,
            passive_ip=passive_ip_,
            passive_port=passive_port_,
        )
        for active_ip_, active_port_, passive_ip_, passive_port_ in zip(
            active_ips, active_ports, passive_ips, passive_ports
        )
    ]
    _logger = logger_factory(role=Const.ACTIVE_NAME)

    # get sample IDs
    _ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

    # start the RSA-Blind-Signature protocol
    if args.phase == "offline":
        _crypto = RSA()
        active_party = ActiveRSAPSI(
            messengers=_messengers,
            cryptosystem=_crypto,
            logger=_logger,
            num_workers=os.cpu_count(),
        )
        active_party.run_offline(_ids)

    elif args.phase == "online":
        _crypto = RSA.from_private_key()
        active_party = ActiveRSAPSI(
            messengers=_messengers,
            cryptosystem=_crypto,
            logger=_logger,
            num_workers=os.cpu_count(),
        )
        active_party.run_online(_ids)

    else:
        raise ValueError(
            "command line argument `--phase` can only"
            "take `offline` and `online`, "
            f"but {args.phase} got instead"
        )

    # close messengers
    for msger in _messengers:
        msger.close()
