import os
import random

from data_processing.config import (
    OUTPUT_DIR,
    TRAIN_SPLIT,
    TRAIN_VAL_SPLIT_RATIO,
    TRAIN_VAL_SPLIT_SEED,
    VAL_SPLIT,
)


def assign_splits(source_keys) -> dict:
    # Sorting first keeps CSV row order and os.listdir order from leaking into
    # which sources end up in val.
    keys = sorted(source_keys)
    if not keys:
        raise ValueError("No source images to split")

    # A dedicated RNG so the split does not consume the global stream that
    # extract_sample draws its rotations and flips from.
    random.Random(TRAIN_VAL_SPLIT_SEED).shuffle(keys)
    n_train = int(len(keys) * TRAIN_VAL_SPLIT_RATIO)

    return {
        key: (TRAIN_SPLIT if ix < n_train else VAL_SPLIT)
        for ix, key in enumerate(keys)
    }


def domain_dir(split: str, domain: str) -> str:
    return os.path.join(OUTPUT_DIR, split, domain)
