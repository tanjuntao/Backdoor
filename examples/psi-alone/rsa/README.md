## RSA Blind Signature based PSI protocol within two parties

### Preparation

#### Active party

Create a new Python file named `alice.py` and fill with the following code in it.

```python 
# active.py
import argparse

from linkefl.crypto import RSACrypto
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocketMessenger
from linkefl.psi.rsa import RSAPSIActive


# set parameters
active_size = 10000
active_ip = 'localhost'
active_port = 10001
passive_ip = 'localhost'
passive_port = 10002
key_size = 1024

# generate dummy ids
active_ids = gen_dummy_ids(size=active_size, option='random')

# initialize messenger 
active_socket = FastSocketMessenger(role='active_party',
                                    active_ip=active_ip,
                                    active_port=active_port,
                                    passive_ip=passive_ip,
                                    passive_port=passive_port)

# create command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str)
args = parser.parse_args()

# start the rsa based psi protocol
if args.phase == 'offline':
    # initialize rsa cryptosystem from scratch
    rsa_crypto = RSACrypto(key_size=key_size)
    # initialize rsa based psi protocol
    active_party = RSAPSIActive(ids=active_ids, 
                                messenger=active_socket, 
                                cryptosystem=rsa_crypto)
    # start the offline phase
    active_party.run_offline()

elif args.phase == 'online':
    # initialize rsa cryptosystem from existing private key, which is generated 
    # at the offline phase
    rsa_crypto = RSACrypto.from_private()
    # initialize rsa based psi protocol
    active_party = RSAPSIActive(ids=active_ids, 
                                messenger=active_socket, 
                                cryptosystem=rsa_crypto)
    # start the online phase
    active_party.run_online()

else:
    raise ValueError(f"command line argument `--phase` can only"
                        f"take `offline` and `online`, "
                        f"but {args.phase} got instead")

# close messenger
active_socket.close()

```

#### Passive party

Create a Python file `passive.py` and fill with the following code in it.

``` python
# passive.py
import argparse

from linkefl.common.const import Const
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocketMessenger
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
    _messenger = FastSocketMessenger(role=Const.PASSIVE_NAME,
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
```

### Run

1. Open one terminal for active party and another one for passive party.

2. Run the offline phase:
    * At the active party terminal, run
    ```shell script
    $ python3 active.py --phase=offline
    ```
    * At the passive party terminal, run
    ```shell script
    $ python3 passive.py --phase=offline
    ```

3. Run the online phase:
    * First, at the active party terminal, run
    ```shell script
    $ python3 active.py --phase=online
    ```
    * Second, at the passive party terminal, run
    ```shell script
    $ python3 passive.py --phase=online
    ```

4. Get the size of the intersection from both the terminal at 
the active party and the terminal at the passive party. 