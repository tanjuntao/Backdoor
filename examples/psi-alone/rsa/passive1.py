import os

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi import PassiveRSAPSI

if __name__ == "__main__":
    # get sample IDs
    passive_ids = gen_dummy_ids(size=100000, option=Const.SEQUENCE)

    active_ip = "127.0.0.1"
    active_port = 20000
    passive_ip = "127.0.0.1"
    passive_port = 30000
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip=active_ip,
        active_port=active_port,
        passive_ip=passive_ip,
        passive_port=passive_port,
    )

    _logger = logger_factory(role=Const.PASSIVE_NAME)
    passive_party = PassiveRSAPSI(
        messenger=_messenger,
        logger=_logger,
        num_workers=os.cpu_count(),
    )
    intersections_ = passive_party.run(passive_ids)
    print(f"length of intersection: {len(intersections_)}")
    _messenger.close()
