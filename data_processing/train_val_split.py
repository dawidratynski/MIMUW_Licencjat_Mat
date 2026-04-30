import os
import random
import shutil

from data_processing.config import TRAIN_VAL_SPLIT_RATIO


def train_val_split(dir):
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    if not files:
        raise ValueError("No files found")

    # Shuffle files
    random.shuffle(files)

    # Split index
    split_idx = int(len(files) * TRAIN_VAL_SPLIT_RATIO)

    train_files = files[:split_idx]
    val_files = files[split_idx:]

    # Create output directories
    train_dir = os.path.join(dir, "train")
    val_dir = os.path.join(dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for f in train_files:
        src = os.path.join(dir, f)
        dst = os.path.join(train_dir, f)
        shutil.move(src, dst)

    for f in val_files:
        src = os.path.join(dir, f)
        dst = os.path.join(val_dir, f)
        shutil.move(src, dst)

    print(
        f"Dataset splitting complete for{dir}: {len(train_files)} train, {len(val_files)} val."
    )
