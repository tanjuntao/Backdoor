from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi.cm20 import CM20PSIActive


if __name__ == '__main__':
    # 1. get sample IDs
    _ids = gen_dummy_ids(size=10_000_000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger1 = FastSocket(
        role=Const.ACTIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20000,
        passive_ip="127.0.0.1",
        passive_port=30000,
    )
    _messenger = [_messenger1]
    _logger = logger_factory(role=Const.ACTIVE_NAME)

    # 3. Start the CM20 protocol
    active_party = CM20PSIActive(_messenger, _logger)
    intersections_ = active_party.run(_ids)
    print(intersections_[:10])

    # 4. Close messenger
    for msger in _messenger:
        msger.close()
