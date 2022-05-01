# active.py
import argparse

from linkefl.common.const import Const
from linkefl.crypto import RSACrypto
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocketMessenger
from linkefl.psi.rsa import RSAPSIActive


# Initialize command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str)
args = parser.parse_args()

# 1. get sample IDs
_ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

# 2. Initialize messenger
_messenger = FastSocketMessenger(role=Const.ACTIVE_NAME,
                                 active_ip='127.0.0.1',
                                 active_port=20001,
                                 passive_ip='127.0.0.1',
                                 passive_port=30001)

# 3. Start the RSA-Blind-Signature protocol
if args.phase == 'offline':
    _crypto = RSACrypto()
    bob = RSAPSIActive(_ids, _messenger, _crypto)
    bob.run_offline()

elif args.phase == 'online':
    _crypto = RSACrypto.from_private()
    bob = RSAPSIActive(_ids, _messenger, _crypto)
    bob.run_online()

else:
    raise ValueError(f"command line argument `--phase` can only"
                     f"take `offline` and `online`, "
                     f"but {args.phase} got instead")

# 4. close messenger
_messenger.close()
