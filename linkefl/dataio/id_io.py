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


def gen_dummy_ids(size, option=Const.SEQUENCE, largest=pow(2, 512), seed=None):
    if option not in (Const.RANDOM, Const.SEQUENCE):
        raise ValueError("option can only be taken from random and sequence, "
                         "but got {} instead".format(option))
    if size > largest:
        raise ValueError("size should not be larger than largetst.")

    if option == Const.SEQUENCE:
        ids = [i for i in range(1, size + 1)] # starting from 1
    else:
        if seed is not None:
            random.seed(seed)
        ids = set()
        while len(ids) != size:
            ids.add(random.randrange(1, largest))

    return ids


if __name__ == '__main__':
    ids_ = gen_dummy_ids(10000000, option=Const.RANDOM, largest=1 << 30, seed=0)
    print(len(ids_))