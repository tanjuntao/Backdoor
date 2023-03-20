from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi import ActiveCM20PSI

if __name__ == "__main__":
    # Get sample IDs
    _ids = gen_dummy_ids(size=1_000_000, option=Const.SEQUENCE)

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

    # Initialize messenger
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

    # Start the CM20 protocol
    active_party = ActiveCM20PSI(messengers=_messengers, logger=_logger)
    intersections_ = active_party.run(_ids)
    print(f"length of intersection: {len(intersections_)}")

    # Close messengers
    for msger in _messengers:
        msger.close()
