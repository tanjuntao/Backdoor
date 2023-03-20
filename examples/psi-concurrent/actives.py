import multiprocessing
import os

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import RSA
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi import ActiveRSAPSI


def task(port_pair):
    active_port, passive_port = port_pair
    _ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

    _messenger1 = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip="127.0.0.1",
        active_port=active_port,
        passive_ip="127.0.0.1",
        passive_port=passive_port,
    )

    _messengers = [_messenger1]
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    _crypto = RSA()

    active_party = ActiveRSAPSI(
        messengers=_messengers,
        cryptosystem=_crypto,
        logger=_logger,
        num_workers=os.cpu_count(),
    )
    intersections_ = active_party.run(_ids)
    print(len(intersections_))

    for msger_ in _messengers:
        msger_.close()


if __name__ == "__main__":
    n = 10
    active_ports = [10000 + i for i in range(n)]
    passive_ports = [20000 + i for i in range(n)]
    with multiprocessing.Pool(5) as pool:
        pool.map(task, zip(active_ports, passive_ports))
        pool.close()
        pool.join()
