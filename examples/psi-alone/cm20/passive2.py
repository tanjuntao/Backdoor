from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi import PassiveCM20PSI

if __name__ == "__main__":
    # Get sample IDs
    _ids = gen_dummy_ids(size=1_000_000, option=Const.SEQUENCE)

    active_ip = "127.0.0.1"
    active_port = 20001
    passive_ip = "127.0.0.1"
    passive_port = 30001

    # Initialize messenger
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )
    _logger = logger_factory(role=Const.PASSIVE_NAME)

    # Start the CM20 protocol
    passive_party = PassiveCM20PSI(messenger=_messenger, logger=_logger)
    intersections_ = passive_party.run(_ids)
    print(f"length of intersection: {len(intersections_)}")

    # Close messenger
    _messenger.close()
