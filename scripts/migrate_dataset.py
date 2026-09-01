'''Migrate an already-generated old-layout dataset into the UVCGAN hierarchy.

`process_data.py` writes crops straight into
`<BASE_DIR>/data/crystal_dataset/{train,val}/{real,synth}` and decides the
train/val split per source photograph before any crop is written. That is the
supported path, and it needs the raw inputs.

This script is the fallback for when the raw inputs are gone and the only
surviving artefact is a dataset built by the old pipeline, laid out as
`<src>/{real,synth}/{train,val}`. It rewrites that tree into the new hierarchy
and re-splits at the source-photograph level in the same pass.

The old split shuffled individual crop files, but the pipeline emits several
crops per source photo (`REAL_SAMPLES_PER_IMAGE`), so siblings of a "held-out"
image ended up in training. Measured on this dataset: 1969 of 1970 real val
sources also appeared in train. That inflates every val metric, and it
specifically breaks the memorisation check -- a val image whose siblings were
trained on looks memorised whether or not the model memorised anything, so the
number stops meaning what it should. The incoming split is therefore discarded
outright: both old split dirs are pooled per domain and re-split from scratch.

Copying is the default, not moving. The old dataset is the only copy of this
data left and cannot be regenerated without the raw inputs, so a run that dies
halfway must not be able to leave it in pieces. Pass --move only once you have
a backup, or once a copy run has already succeeded.

    python scripts/migrate_dataset.py dataset ../data/crystal_dataset --dry_run
    python scripts/migrate_dataset.py dataset ../data/crystal_dataset
'''
import argparse
import os
import random
import re
import shutil
from collections import defaultdict

SAMPLE_SUFFIX = re.compile(r'_sample_\d+\.png$')
DOMAINS = ('real', 'synth')
SPLITS = ('train', 'val')


def source_key(filename: str) -> str:
    stripped = SAMPLE_SUFFIX.sub('', filename)
    if stripped == filename:
        # No _sample_N suffix: treat the file as its own source rather than
        # silently lumping every such file into one giant group.
        return filename
    return stripped


def collect(domain_root: str) -> dict:
    groups = defaultdict(list)
    for split in SPLITS:
        d = os.path.join(domain_root, split)
        if not os.path.isdir(d):
            # The old split is discarded anyway, so a split dir that was
            # emptied or never existed costs nothing -- only both being gone
            # is fatal, and that surfaces as an empty group set below.
            print(f'  note: {d} does not exist, skipping')
            continue
        for fn in os.listdir(d):
            if fn.endswith('.png'):
                groups[source_key(fn)].append((split, fn))
    return groups


def leakage(groups: dict) -> int:
    return sum(1 for files in groups.values()
               if len({s for s, _ in files}) > 1)


def assign_splits(source_keys, ratio: float, seed: int) -> dict:
    # Mirrors data_processing/split.py::assign_splits. Reimplemented rather
    # than imported: that version reads the ratio and seed from
    # data_processing.config as module constants, so it cannot honour
    # --ratio/--seed, and importing it would tie this one-off migration to a
    # BASE_DIR the user has no way to set correctly once the raw data is gone.
    keys = sorted(source_keys)
    if not keys:
        raise ValueError('No source images to split')

    random.Random(seed).shuffle(keys)
    n_train = int(len(keys) * ratio)

    return {key: ('train' if ix < n_train else 'val')
            for ix, key in enumerate(keys)}


def verify(dst_root: str) -> None:
    for domain in DOMAINS:
        seen = defaultdict(set)
        for split in SPLITS:
            d = os.path.join(dst_root, split, domain)
            for fn in os.listdir(d):
                if fn.endswith('.png'):
                    seen[source_key(fn)].add(split)
        leaked = [k for k, splits in seen.items() if len(splits) > 1]
        if leaked:
            raise RuntimeError(
                f'{domain}: {len(leaked)} sources still span train and val, '
                f'e.g. {leaked[:3]}')


def main():
    p = argparse.ArgumentParser(
        description='Migrate an old-layout dataset (real/{train,val}, '
                    'synth/{train,val}) into the UVCGAN hierarchy '
                    '({train,val}/{real,synth}), re-splitting per source '
                    'photograph so crops of one photo never straddle the '
                    'split. Copies by default, leaving the original intact, '
                    'because it is the only copy of this data and cannot be '
                    'regenerated without the raw inputs.')
    p.add_argument('src_dataset_dir',
                   help='Old dataset root, containing real/ and synth/, each '
                        'with train/ and val/.')
    p.add_argument('dst_crystal_dataset_dir',
                   help='New crystal_dataset root. train/{real,synth} and '
                        'val/{real,synth} are created underneath it.')
    p.add_argument('--ratio', type=float, default=0.8,
                   help='Fraction of source photographs used for train.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--move', action='store_true',
                   help='Move files instead of copying them. Destructive: it '
                        'consumes the only copy of the dataset, so use it '
                        'only with a backup or after a copy run succeeded.')
    p.add_argument('--dry_run', action='store_true',
                   help='Print the plan and write nothing.')
    args = p.parse_args()

    src, dst = args.src_dataset_dir, args.dst_crystal_dataset_dir
    if not os.path.isdir(src):
        raise FileNotFoundError(f'{src} does not exist')

    transfers = []
    landing = {(s, d): 0 for s in SPLITS for d in DOMAINS}

    for domain in DOMAINS:
        domain_root = os.path.join(src, domain)
        if not os.path.isdir(domain_root):
            raise FileNotFoundError(f'{domain_root} does not exist')

        print(f'\n{domain}')
        groups = collect(domain_root)
        if not groups:
            raise FileNotFoundError(f'No .png files found under {domain_root}')

        old_counts = defaultdict(int)
        for files in groups.values():
            for split, _ in files:
                old_counts[split] += 1
        n_files = sum(len(v) for v in groups.values())
        leaked = leakage(groups)

        print(f'  before: {n_files} files, {len(groups)} sources '
              f'({old_counts["train"]} train, {old_counts["val"]} val)')
        print(f'  sources split across the old train and val: {leaked}'
              f'{"  <-- leakage" if leaked else ""}')

        target = assign_splits(groups, args.ratio, args.seed)
        for key, files in groups.items():
            want = target[key]
            landing[(want, domain)] += len(files)
            for cur, fn in files:
                transfers.append((os.path.join(domain_root, cur, fn),
                                  os.path.join(dst, want, domain, fn)))

        n_train_src = sum(1 for v in target.values() if v == 'train')
        print(f'  after:  {n_train_src} sources / '
              f'{landing[("train", domain)]} files train, '
              f'{len(groups) - n_train_src} sources / '
              f'{landing[("val", domain)]} files val')

    verb = 'move' if args.move else 'copy'
    print(f'\n{len(transfers)} files to {verb} into {dst}')
    for split in SPLITS:
        for domain in DOMAINS:
            print(f'  {split}/{domain}: {landing[(split, domain)]}')

    if args.dry_run:
        print('\ndry run, nothing written')
        return

    for split in SPLITS:
        for domain in DOMAINS:
            os.makedirs(os.path.join(dst, split, domain), exist_ok=True)

    transfer = shutil.move if args.move else shutil.copy2
    for ix, (src_path, dst_path) in enumerate(transfers, 1):
        transfer(src_path, dst_path)
        if ix % 2000 == 0:
            print(f'  {ix}/{len(transfers)}')

    verify(dst)
    print('\ndone, no source spans train and val')


if __name__ == '__main__':
    main()
