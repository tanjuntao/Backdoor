import argparse
import os

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi import PassiveRSAPSI

# 1. If you are on macOS and use a python version higher than 3.8, you should
#    wrap the multiprocessing code inside the __main__ clause, because on macOS
#    after python 3.8, it used **spawn** as the default method to generate
#    subprocess, and this requires the main module to be safe imported;
# 2. If you are on Linux, then you need not wrap the multiprocessing code
#    inside the __main__ clause, which means that you can use it directly inside
#    your python module. This is because than on Linux python use **fork** as
#    the default method to generate subprocess.
if __name__ == "__main__":
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str)
    args = parser.parse_args()

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

    # Get sample IDs
    _ids = gen_dummy_ids(size=100000, option=Const.SEQUENCE)

    # Start the RSA-Blind-Signature protocol
    passive_party = PassiveRSAPSI(
        messenger=_messenger, logger=_logger, num_workers=os.cpu_count()
    )
    if args.phase == "offline":
        passive_party.run_offline(_ids)
    elif args.phase == "online":
        passive_party.run_online(_ids)
    else:
        raise ValueError(
            "command line argument `--phase` can only"
            "take `offline` and `online`, "
            f"but {args.phase} got instead"
        )

    # 4. close messenger
    _messenger.close()
