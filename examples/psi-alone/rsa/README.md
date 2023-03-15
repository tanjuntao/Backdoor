## RSA Blind Signature based PSI protocol within two parties

### Preparation

#### Active party

Create a new Python file named `active_phase.py` and fill with the following code in it.

```python
import argparse
import os

from linkefl.common.const import Const
from linkefl.common.factory import logger_factory
from linkefl.crypto import RSA
from linkefl.dataio import gen_dummy_ids
from linkefl.messenger import FastSocket
from linkefl.psi import ActiveRSAPSI


if __name__ == "__main__":
    # Initialize command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str)
    args = parser.parse_args()

    active_ips = ["127.0.0.1", ]
    active_ports = [20000, ]
    passive_ips = ["127.0.0.1", ]
    passive_ports = [30000, ]
    _messengers = [
        FastSocket(
            role=Const.ACTIVE_NAME,
            active_ip=active_ip_,
            active_port=active_port_,
            passive_ip=passive_ip_,
            passive_port=passive_port_,
        )
        for active_ip_, active_port_, passive_ip_, passive_port_ in
        zip(active_ips, active_ports, passive_ips, passive_ports)
    ]
    _logger = logger_factory(role=Const.ACTIVE_NAME)

    # get sample IDs
    _ids = gen_dummy_ids(size=10000, option=Const.SEQUENCE)

    # start the RSA-Blind-Signature protocol
    if args.phase == "offline":
        _crypto = RSA()
        active_party = ActiveRSAPSI(
            messengers=_messengers,
            cryptosystem=_crypto,
            logger=_logger,
            num_workers=os.cpu_count(),
        )
        active_party.run_offline(_ids)

    elif args.phase == "online":
        _crypto = RSA.from_private_key()
        active_party = ActiveRSAPSI(
            messengers=_messengers,
            cryptosystem=_crypto,
            logger=_logger,
            num_workers=os.cpu_count(),
        )
        active_party.run_online(_ids)

    else:
        raise ValueError(
            "command line argument `--phase` can only"
            "take `offline` and `online`, "
            f"but {args.phase} got instead"
        )

    # close messengers
    for msger in _messengers:
        msger.close()


```
#### Passive party

Create a Python file `passive_phase.py` and fill with the following code in it.

```python
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
        messenger=_messenger,
        logger=_logger,
        num_workers=os.cpu_count()
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


```
### Run

1. Open one terminal for active party and another one for passive party.

2. Run the offline phase:
    * At the active party terminal, run
    ```shell script
    $ python3 active_phase.py --phase=offline
    ```
    * At the passive party terminal, run
    ```shell script
    $ python3 passive_phase.py --phase=offline
    ```

3. Run the online phase:
    * First, at the active party terminal, run
    ```shell script
    $ python3 active_phase.py --phase=online
    ```
    * Second, at the passive party terminal, run
    ```shell script
    $ python3 passive_phase.py --phase=online
    ```

4. Get the size of the intersection from both the terminal at
the active party and the terminal at the passive party.
