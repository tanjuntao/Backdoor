import random
import warnings

from tqdm import trange

from linkefl.common.const import Const


def get_ids(role, config, largest_val=1 << 512):
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


def gen_dummy_ids(size,
                  option=Const.SEQUENCE,
                  largest=1 << 512,
                  unique=False,
                  seed=None):
    if option not in (Const.RANDOM, Const.SEQUENCE):
        raise ValueError("option can only be taken from random and sequence, "
                         "but got {} instead".format(option))
    if size > largest:
        raise ValueError("size should not be larger than largetst.")

    print('Generating dummy ids...')
    if option == Const.SEQUENCE:
        print('Done!')
        ids = [i for i in range(1, size + 1)] # starting from 1
        return ids

    if seed is not None:
        random.seed(seed)
    if unique:
        warnings.warn('This process may take a REALLY long time. Be careful using it.')
        if largest <= (1 << 63) - 1:
            ids = random.sample(range(largest), size)
        else:
            # This branch will introduce Python set data structure. Getting items from
            # a big set and deleting a big set is REALLY slow. Be careful using it.
            expand_factor = 1.02
            new_step = 10_0000
            expanded_size = int(expand_factor * size)
            expanded_ids = [random.randrange(1, largest) for _ in range(expanded_size)]
            set_ids = set(expanded_ids) # remove duplicated items
            del expanded_ids # save memory, del operation on list is efficient
            while len(set_ids) < size:
                new_ids = [random.randrange(1, largest) for _ in range(new_step)]
                set_ids.update(new_ids)
                del new_ids # save memory
            ids = []
            for item in set_ids:
                ids.append(item)
                if len(ids) == size:
                    break
    else:
        ids = [random.randrange(1, largest) for _ in trange(size)]
    print('Done!')
    return ids


if __name__ == '__main__':
    import time
    start_time = time.time()
    ids_ = gen_dummy_ids(
        size=10_000_000,
        option=Const.RANDOM,
        unique=True,
        largest=1 << 32,
        seed=0
    )
    print(time.time() - start_time)
    print(len(ids_))