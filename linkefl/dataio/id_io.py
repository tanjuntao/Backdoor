import random

from linkefl.common.const import Const


def get_ids(role, config, largest_val=pow(2, 512)):
    """Get RSAPSIPassive's IDs and RSAPSIActive's IDs

    Args:
        role: Party role name.
        config: configuration object
        largest_val: The maximum value that an int ID can take.

    Returns:
        party ids: A python list.
    """
    if role == config.ROLE_ALICE:
        # ids = [random.SystemRandom().randrange(1, largest_val)
        #        for _ in range(Config.ALICE_SIZE)]
        ids = [i for i in range(config.ALICE_SIZE)]

    elif role == config.ROLE_BOB:
        # ids = [random.SystemRandom().randrange(1, largest_val)
        #        for _ in range(Config.BOB_SIZE)]
        ids = [i for i in range(config.BOB_SIZE)]

    else:
        raise ValueError(f"role can only take 'alice' and 'bob',"
                         f"but got {role} instead.")
    return ids


def gen_dummy_ids(size, option=Const.SEQUENCE, largest=pow(2, 512)):
    assert option in (Const.RANDOM, Const.SEQUENCE), \
        'option can only take `random` and `sequence`, ' \
        'but {} got instead'.format(option)

    if option == Const.SEQUENCE:
        ids = [i for i in range(size)]
    else:
        ids = [random.SystemRandom().randrange(1, largest) for _ in range(size)]

    return ids


