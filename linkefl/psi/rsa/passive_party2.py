import argparse
import os

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi import PassiveRSAPSI

if __name__ == "__main__":
    #   Option 1: split the protocol
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str)
    args = parser.parse_args()

    # # 1. Get sample IDs
    # _ids = gen_dummy_ids(size=100000, option=Const.SEQUENCE)
    #
    # # 2. Initialize messenger
    # _messenger = FastSocket(role=Const.PASSIVE_NAME,
    #                         active_ip='127.0.0.1',
    #                         active_port=20000,
    #                         passive_ip='127.0.0.1',
    #                         passive_port=30000)
    # _logger = logger_factory(role=Const.ACTIVE_NAME)
    #
    # # 3. Start the RSA-Blind-Signature protocol
    # alice = PassiveRSAPSI(_messenger, _logger)
    # if args.phase == 'offline':
    #     alice.run_offline(_ids)
    # elif args.phase == 'online':
    #     alice.run_online(_ids)
    # else:
    #     raise ValueError(f"command line argument `--phase` can only"
    #                      f"take `offline` and `online`, "
    #                      f"but {args.phase} got instead")
    #
    # # 4. close messenger
    # _messenger.close()

    # '''
    #   Option 2: run the whole protocol
    # 1. Get sample IDs
    _ids = gen_dummy_ids(size=1000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocket(
        role=Const.PASSIVE_NAME,
        active_ip="127.0.0.1",
        active_port=20001,
        passive_ip="127.0.0.1",
        passive_port=30001,
    )
    _logger = logger_factory(role=Const.ACTIVE_NAME)
    # 3. Start the RSA-Blind-Signature protocol
    passive_party = PassiveRSAPSI(
        messenger=_messenger, logger=_logger, num_workers=os.cpu_count()
    )
    intersections_ = passive_party.run(_ids)

    # 4. Close messenger
    _messenger.close()
    # '''
