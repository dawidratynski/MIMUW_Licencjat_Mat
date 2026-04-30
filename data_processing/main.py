import random
import shutil
import os
import logging

import numpy as np

from data_processing.config import *
from data_processing.real_images import process_real_images
from data_processing.synth_images import process_synth_images
from data_processing.train_val_split import train_val_split


def prepare_folders() -> None:
    target_dir = OUTPUT_DIR
    if os.path.exists(target_dir):
        if REMOVE_OUTPUT_DIR_IF_PRESENT:
            shutil.rmtree(target_dir)
        else:
            raise ValueError(f"Target directory {target_dir} already exists")

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_dir + "/real", exist_ok=True)
    os.makedirs(target_dir + "/synth", exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    random.seed(42)
    np.random.seed(42)

    prepare_folders()

    process_real_images()
    process_synth_images()

    train_val_split(os.path.join(OUTPUT_DIR, "real"))
    train_val_split(os.path.join(OUTPUT_DIR, "synth"))


if __name__ == "__main__":
    main()
