from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi.cm20 import CM20PSIPassive


if __name__ == "__main__":
    # 1. Get sample IDs
    _ids = gen_dummy_ids(size=10_000_000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20000,
        passive_ip="127.0.0.1",
        passive_port=30000,
    )
    _logger = logger_factory(role=Const.PASSIVE_NAME)

    # 3. Start the CM20 protocol
    passive_party = CM20PSIPassive(_messenger, _logger)
    intersections_ = passive_party.run(_ids)
    print(intersections_[:10])

    # 4. Close messenger
    _messenger.close()