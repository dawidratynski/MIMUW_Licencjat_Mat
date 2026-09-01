import os
import numpy as np
import PIL.Image

from collections import defaultdict

from PIL.Image import Image

from logging import warn, info, debug

from data_processing.config import *
from data_processing.sample import extract_sample
from data_processing.split import assign_splits, domain_dir


def resolve_synth_data_files() -> list[str]:
    if not os.path.isdir(SYNTH_DATA_DIR):
        raise FileNotFoundError(
            f"Synth image directory not found: {SYNTH_DATA_DIR}. It must hold the "
            f".npy stacks of synthetic images. Point SYNTH_DATA_DIR in "
            f"data_processing/config.py at the directory holding them."
        )

    # An explicit SYNTH_DATA_FILES is authoritative and every entry must exist
    # and be a 3-D stack, so a missing or wrong-shaped file aborts the run
    # rather than silently yielding a smaller dataset. Discovery is the fallback
    # for an empty list: params companions and any other non-3-D .npy are
    # skipped with a warning, and the remainder is sorted so the train/val
    # split stays reproducible across machines.
    if SYNTH_DATA_FILES:
        for filename in SYNTH_DATA_FILES:
            if not os.path.isfile(os.path.join(SYNTH_DATA_DIR, filename)):
                raise FileNotFoundError(
                    f"Synth stack {filename!r} listed in SYNTH_DATA_FILES is missing "
                    f"from {SYNTH_DATA_DIR}. Fix the name, restore the file, or set "
                    f"SYNTH_DATA_FILES = [] in data_processing/config.py to use "
                    f"every 3-D .npy image stack found in that directory."
                )
        return list(SYNTH_DATA_FILES)

    candidates = sorted(
        entry for entry in os.listdir(SYNTH_DATA_DIR) if entry.endswith(".npy")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .npy stacks found in {SYNTH_DATA_DIR}. Point SYNTH_DATA_DIR in "
            f"data_processing/config.py at the directory holding the synthetic "
            f"image stacks, or list them in SYNTH_DATA_FILES."
        )

    discovered = []
    for filename in candidates:
        if "params" in filename.lower():
            warn(
                f"Skipping {filename!r} in {SYNTH_DATA_DIR}: filename looks like "
                f"a params array, not a 3-D image stack."
            )
            continue

        array = np.load(os.path.join(SYNTH_DATA_DIR, filename), mmap_mode="r")
        if array.ndim != 3:
            warn(
                f"Skipping {filename!r} in {SYNTH_DATA_DIR}: unsupported shape "
                f"{array.shape}. Expected a 3-D stack of (n_images, height, width)."
            )
            continue

        discovered.append(filename)

    if not discovered:
        raise FileNotFoundError(
            f"No 3-D .npy image stacks found in {SYNTH_DATA_DIR}. Companion "
            f"params_*.npy files are ignored. Point SYNTH_DATA_DIR in "
            f"data_processing/config.py at the directory holding the synthetic "
            f"image stacks, or list them in SYNTH_DATA_FILES."
        )

    info(f"Discovered {len(discovered)} synth stacks in {SYNTH_DATA_DIR}")
    return discovered


def stack_lengths(filenames: list[str]) -> list[int]:
    lengths = []

    for filename in filenames:
        images = np.load(os.path.join(SYNTH_DATA_DIR, filename), mmap_mode="r")
        if images.ndim != 3:
            # Explicit SYNTH_DATA_FILES stays strict: a listed file with the
            # wrong ndim is a configuration error. Discovery already skipped
            # non-3-D companions (params_*.npy and any other shape).
            raise ValueError(
                f"Unsupported shape: {images.shape} in {filename!r} "
                f"(from {SYNTH_DATA_DIR}). Expected a 3-D stack of "
                f"(n_images, height, width). If this is not a synthetic image "
                f"stack, remove it from SYNTH_DATA_FILES in "
                f"data_processing/config.py, or leave that list empty to "
                f"discover 3-D stacks and skip the rest."
            )

        lengths.append(len(images))

    return lengths


def allocate_quotas(capacities: list[int], limit: int) -> list[int]:
    # Every stack is a different region of the generator's parameter space
    # (h, k, l, uc_scale_ab, channel_vol_prop), so the quota is split evenly
    # instead of being filled from the first stack. A stack that is too small
    # to meet its share hands the shortfall back to the stacks that still have
    # room, which is why this is a loop rather than a single division.
    quotas = [0] * len(capacities)
    remaining = min(limit, sum(capacities))

    while remaining > 0:
        with_capacity = [ix for ix, cap in enumerate(capacities) if quotas[ix] < cap]
        share, extra = divmod(remaining, len(with_capacity))

        for rank, ix in enumerate(with_capacity):
            wanted = share + (1 if rank < extra else 0)
            take = min(wanted, capacities[ix] - quotas[ix])
            quotas[ix] += take
            remaining -= take

    return quotas


def stride_indices(length: int, quota: int) -> list[int]:
    # An evenly spaced stride rather than the first `quota` frames: the
    # generator may have written each stack in a systematic order (sweeping a
    # parameter monotonically), in which case a leading slice is a biased
    # corner of that stack. Integer linspace over [0, length-1] is deterministic
    # and duplicate-free when quota <= length (allocate_quotas guarantees that),
    # and uses no RNG so the global random stream extract_sample draws from is
    # left untouched.
    if quota <= 0:
        return []
    if quota >= length:
        return list(range(length))
    if quota == 1:
        return [0]

    return [ix * (length - 1) // (quota - 1) for ix in range(quota)]


def collect_synth_sources(filenames: list[str]) -> list[tuple[str, int]]:
    capacities = stack_lengths(filenames)
    quotas = allocate_quotas(capacities, SYNTH_IMAGE_COUNT_LIMIT)

    sources = []
    for filename, length, quota in zip(filenames, capacities, quotas):
        sources.extend(
            (filename, image_ix) for image_ix in stride_indices(length, quota)
        )

    if len(sources) < SYNTH_IMAGE_COUNT_LIMIT:
        warn(
            f"Synth stacks hold only {sum(capacities)} frames in total: requested "
            f"{SYNTH_IMAGE_COUNT_LIMIT}, obtained {len(sources)}. Add stacks to "
            f"{SYNTH_DATA_DIR} or lower SYNTH_IMAGE_COUNT_LIMIT in "
            f"data_processing/config.py to silence this."
        )

    info(
        f"Selected {len(sources)} synth frames evenly across {len(filenames)} "
        f"stacks (per-stack: "
        f"{', '.join(f'{name}={quota}' for name, quota in zip(filenames, quotas))})"
    )

    return sources


def process_synth_images() -> None:
    info("Starting processing of synth images")

    filenames = resolve_synth_data_files()
    sources = collect_synth_sources(filenames)
    splits = assign_splits(sources)

    by_file = defaultdict(list)
    for filename, image_ix in sources:
        by_file[filename].append(image_ix)

    for filename in filenames:
        if filename not in by_file:
            continue

        info(f"Processing file: {filename}")
        images = np.load(os.path.join(SYNTH_DATA_DIR, filename), mmap_mode="r")

        for image_ix in by_file[filename]:
            image_raw = images[image_ix]

            # Ensure correct data type
            if image_raw.dtype != np.uint8:
                warn(
                    f"{filename} @ {image_ix}: Image has wrong dtype: {image_raw.dtype}. Converting to uint8."
                )
                image_raw = np.clip(image_raw, 0, 255).astype(np.uint8)

            image = PIL.Image.fromarray(image_raw)
            target_dir = domain_dir(splits[(filename, image_ix)], "synth")

            debug(f"Processing image {filename} @ {image_ix}")

            for sample_ix in range(SYNTH_SAMPLES_PER_IMAGE):
                sample: Image = extract_sample(image, SYNTH_IMAGE_RESOLUTION_PX_PER_NM)
                output_filename = f"{filename}_@_{image_ix}_sample_{sample_ix}.png"
                sample.save(os.path.join(target_dir, output_filename), "PNG")

    info("Finished processing synth images successfully.")
