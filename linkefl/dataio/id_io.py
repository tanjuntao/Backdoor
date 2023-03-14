import random
import warnings
from typing import List, Optional

from tqdm import trange

from linkefl.common.const import Const


def gen_dummy_ids(
    size: int,
    *,
    option: str = "sequence",
    largest: int = 1 << 512,
    unique: bool = False,
    seed: Optional[int] = None,
) -> List[int]:
    if option not in (Const.RANDOM, Const.SEQUENCE):
        raise ValueError(
            "option can only be taken from random and sequence, "
            "but got {} instead".format(option)
        )
    if size > largest:
        raise ValueError("size should not be larger than largetst.")

    print("Generating dummy ids...")
    if option == Const.SEQUENCE:
        print("Done!")
        ids = [i for i in range(1, size + 1)]  # starting from 1
        return ids

    if seed is not None:
        random.seed(seed)
    if unique:
        warnings.warn("This process may take a REALLY long time. Be careful using it.")
        if largest <= (1 << 63) - 1:
            ids = random.sample(range(largest), size)
        else:
            # This branch will introduce Python set data structure. Getting items from
            # a big set and deleting a big set is REALLY slow. Be careful using it.
            expand_factor = 1.02
            new_step = 10_0000
            expanded_size = int(expand_factor * size)
            expanded_ids = [random.randrange(1, largest) for _ in range(expanded_size)]
            set_ids = set(expanded_ids)  # remove duplicated items
            del expanded_ids  # save memory, del operation on list is efficient
            while len(set_ids) < size:
                new_ids = [random.randrange(1, largest) for _ in range(new_step)]
                set_ids.update(new_ids)
                del new_ids  # save memory
            ids = []
            for item in set_ids:
                ids.append(item)
                if len(ids) == size:
                    break
    else:
        ids = [random.randrange(1, largest) for _ in trange(size)]
    print("Done!")
    return ids


if __name__ == "__main__":
    import time

    start_time = time.time()
    ids_ = gen_dummy_ids(
        size=10_000_000, option=Const.RANDOM, unique=True, largest=1 << 32, seed=0
    )
    print(time.time() - start_time)
    print(len(ids_))
