# passive.py
import argparse

from linkefl.common.const import Const
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi.rsa import RSAPSIPassive


# 1. If you are on macOS and use a python version higher than 3.8, you should
#    wrap the multiprocessing code inside the __main__ clause, because on macOS
#    after python 3.8, it used **spawn** as the default method to generate
#    subprocess, and this requires the main module to be safe imported;
# 2. If you are on Linux, then you need not to wrap the multiprocessing code
#    inside the __main__ clause, which means that you can use it directly inside
#    your python module. This is because than on Linux python use **fork** as
#    the default method to generate subprocess.
if __name__ == '__main__':
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str)
    args = parser.parse_args()

    # 1. Get sample IDs
    _ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

    # 2. Initialize messenger
    _messenger = FastSocket(role=Const.PASSIVE_NAME,
                            active_ip='127.0.0.1',
                            active_port=20001,
                            passive_ip='127.0.0.1',
                            passive_port=30001)

    # 3. Start the RSA-Blind-Signature protocol
    alice = RSAPSIPassive(_ids, _messenger)
    if args.phase == 'offline':
        alice.run_offline()
    elif args.phase == 'online':
        alice.run_online()
    else:
        raise ValueError(f"command line argument `--phase` can only"
                         f"take `offline` and `online`, "
                         f"but {args.phase} got instead")

    # 4. close messenger
    _messenger.close()