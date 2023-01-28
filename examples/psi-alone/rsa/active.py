# active.py
import argparse

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import RSA
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import RSAPSIActive

# Initialize command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str)
args = parser.parse_args()

# 1. get sample IDs
_ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

# 2. Initialize messenger and logger
_messenger1 = FastSocket(
    role=Const.ACTIVE_NAME,
    active_ip="127.0.0.1",
    active_port=20001,
    passive_ip="127.0.0.1",
    passive_port=30001,
)
_messenger = [_messenger1]
_logger = logger_factory(role=Const.ACTIVE_NAME)

# 3. Start the RSA-Blind-Signature protocol
if args.phase == "offline":
    _crypto = RSA()
    bob = RSAPSIActive(_messenger, _crypto, _logger)
    bob.run_offline(_ids)

elif args.phase == "online":
    _crypto = RSA.from_private_key()
    bob = RSAPSIActive(_messenger, _crypto, _logger)
    bob.run_online(_ids)

else:
    raise ValueError(
        "command line argument `--phase` can only"
        "take `offline` and `online`, "
        f"but {args.phase} got instead"
    )

# 4. close messenger
for msger in _messenger:
    msger.close()
