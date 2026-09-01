# MIMUW Licencjat — UVCGAN2 on crystal lattice images

Trains UVCGAN2 to estimate both translations between two image domains:

- `synth` (S) — synthetic crystal-lattice images unpacked from `.npy` stacks
- `real` (R) — real TEM crystal photographs

A single UVCGAN2 run produces both generators: `gen_ab` = A: S -> R and
`gen_ba` = B: R -> S. `gen_ba` is the forward operator used by the sibling
`diffusion-posterior-sampling` repository.

Training is two-staged, following upstream UVCGAN2: a BERT-style masked
generator pretraining, then the actual translation training that transfers the
pretrained encoder weights into `gen_ab` and `gen_ba`.

## Requirements

Install `torch` and `torchvision` yourself first. The vendored `uvcgan2`
package does not declare them, and the correct wheel depends on your CUDA
version — use the selector at <https://pytorch.org/get-started/locally/>.

Then install the remaining dependencies, and only afterwards run the step 1
setup script:

```powershell
pip install -r requirements.txt
```

`requirements.txt` pins `pandas<2.0`. The vendored training-history callback
calls `DataFrame.append`, which pandas 2.0 removed, so with a newer pandas both
training stages crash at the end of epoch 2. Do not unpin it.

## Run order

### 1. Install the vendored uvcgan2 (once)

```powershell
python uvcgan_training/setup_uvcgan.py
```

`uvcgan_training/UVCGAN/source` is already vendored, so this only runs
`pip install -e` on it.

### 2. Point the pipeline at your data

Set `BASE_DIR` in `data_processing/config.py` (it is a Python constant, not a
shell variable). That file keeps the two locations separate:

- `RAW_DIR` — where the raw inputs are read from. It defaults to `daneProsem/`
  at the repository root and must contain `crop_filtering_by_region/`,
  `data_summary_2957.csv` and the `.npy` stacks.
- `BASE_DIR` — where the generated dataset is written. Keep it off the
  Dropbox-synced repository tree: the pipeline emits roughly 30k PNGs, and
  syncing them is slow and pointless.

`UVCGAN_DATA_DIR` is `BASE_DIR/data`. The dataset itself is written to
`OUTPUT_DIR` = `UVCGAN_DATA_DIR/crystal_dataset`.

The expected raw layout is

```
daneProsem/
    data_summary_2957.csv          metadata for the real domain
    Kopia metadata.json            generator settings for the synth domain
    *.npy                          synthetic image stacks (and optional params companions)
    crop_filtering_by_region/      per-object TEM JPEGs, one per CSV row
```

`crop_filtering_by_region/` (`REAL_DATA_DIR`) holds the files named in the CSV's
`name_PLB` column, for example `123_oat_ETIO_twoweeks_30kx_(1)_obj2.jpg`. The
`.npy` stacks sit at the top level of `daneProsem/` rather than in a
subdirectory, so `SYNTH_DATA_DIR` is `RAW_DIR` itself. Files named in the CSV but
missing from the crop directory are logged and skipped, so a partial crop
directory runs but silently shrinks the dataset; a missing listed `.npy` is fatal.

`SYNTH_DATA_FILES` defaults to `[]`, which means every 3-D `*.npy` image stack in
`SYNTH_DATA_DIR` is used, sorted by name. Companion `params_*.npy` files next to
the image stacks are ignored, as is any other non-3-D `.npy` (skipped with a
warning that names the file and shape). Listing the stacks explicitly is
stricter and worth doing once their names are known, because then a stack that
disappears or has the wrong ndim aborts the run instead of quietly producing a
smaller dataset. `SYNTH_IMAGE_COUNT_LIMIT` frames are drawn evenly across all
discovered image stacks (equal per-stack quota, remainder to the first stacks),
and within each stack an even stride over the full length is used rather than a
leading slice.

### 3. Build the dataset

Run from the repository root so that the `data_processing` package resolves:

```powershell
python process_data.py
```

This writes `$BASE_DIR/data/crystal_dataset/{train,val}/{real,synth}`.

If the JPEG / `.npy` sources in `daneProsem/` are gone and all you have is a
dataset built by the old pipeline — laid out as
`dataset/{real,synth}/{train,val}` — then `process_data.py` cannot run, and
you migrate that tree instead:

```powershell
python scripts/migrate_dataset.py <old-layout dataset root> <OUTPUT_DIR> --dry_run
```

Drop `--dry_run` to write; the destination must be `OUTPUT_DIR` from
`data_processing/config.py` (`BASE_DIR/data/crystal_dataset`).
It copies by default and leaves the old tree untouched, and it re-splits per
object crop, which the old pipeline did not do. A migrated dataset
is not bit-identical to a freshly generated one, and it is also weaker: the fresh
pipeline keys the real-domain split on the CSV's `raw_png` column, so it groups
at the level of the source photograph, while migration can only see output
filenames and therefore groups at the level of the object crop. Both are valid
80/20 splits, but only the freshly generated one is free of the
same-photograph correlation described under [Data layout](#data-layout).

### 4. Set the UVCGAN2 environment variables

`UVCGAN2_DATA` is the parent of `crystal_dataset` (not `crystal_dataset`
itself). It must equal `UVCGAN_DATA_DIR` from `data_processing/config.py`,
i.e. `BASE_DIR/data` — the same directory the processing pipeline writes
through `BASE_DIR` / `UVCGAN_DATA_DIR`. uvcgan2 resolves datasets as
`$UVCGAN2_DATA/<path>` with `path = 'crystal_dataset'`.

`UVCGAN2_OUTDIR` is where training writes checkpoints and models. It should
be absolute: `uvcgan2/consts.py` reads it once at import and resolves it
against the current working directory, and stage 2 looks the stage-1 model
up as `$UVCGAN2_OUTDIR/<base_model>`. With a relative value every subsequent
command must be run from `uvcgan_training/`, otherwise the transfer fails.

```powershell
cd uvcgan_training
$env:UVCGAN2_DATA   = "..."
$env:UVCGAN2_OUTDIR = "..."
```

```bash
cd uvcgan_training
export UVCGAN2_DATA=...
export UVCGAN2_OUTDIR=...
```

### 5. Stage 1 — generator pretraining

```powershell
python pretrain_crystals.py --batch-size 32
```

This is the bulk of the compute. It writes
`$UVCGAN2_OUTDIR/crystal_experiments/model_m(simple-autoencoder)_d(None)_g(vit-modnet)_pretrain-uvcgan2`.

### 6. Stage 2 — translation training

```powershell
python train_crystals.py
```

Trains `gen_ab` (S -> R) and `gen_ba` (R -> S) together, loading the stage-1
encoder. Output goes to `$UVCGAN2_OUTDIR/crystal_experiments/translation/...`.
Pass `--no-pretrain` to skip the transfer and train from scratch; the run label
records which variant was used, so the two do not share a save directory.

### 7. Translate images

```powershell
python UVCGAN/source/scripts/translate_images.py `
    "$env:UVCGAN2_OUTDIR/crystal_experiments/translation/<savedir>" `
    --split val --batch-size 8
```

`--split val` is required. `translate_images.py` defaults to `--split test`,
and this pipeline only produces `train` and `val`.

## Trained models

After stage 2 the models are in

```
$UVCGAN2_OUTDIR/crystal_experiments/translation/model_m(uvcgan2)_d(basic)_g(vit-modnet)_uvcgan2-bsd_(pre_True_cyc_10.0_gp_0.01_lr_0.0001)/
```

which is the literal directory name for a default run; the trailing label
changes with `--no-pretrain`, `--lambda-cycle`, `--lambda-gp` and `--lr-gen`.
It contains:

- `net_gen_ab.pth` — A: synth -> real
- `net_gen_ba.pth` — B: real -> synth
- `net_avg_gen_ab.pth`, `net_avg_gen_ba.pth` — EMA copies of those two,
  momentum 0.9999

Use the `avg_` (EMA) variants for inference. `UVCGAN2.forward` (in
`uvcgan2/cgan/uvcgan2.py`) dispatches to `avg-ab`/`avg-ba` whenever
`avg_momentum` is set, so every uvcgan2 eval script implicitly reports the EMA
generators. Anything downstream — including `diffusion-posterior-sampling` —
must load `avg_gen_ba` to be measuring the same operator.

## Gotchas

Stage 1 must run to completion. `train_crystals.py::get_transfer_preset` sets
`'allow_partial' : False`, so a stage-1 run killed partway — one that has
numbered files under `checkpoints/` but no final `net_encoder.pth` — makes
stage 2 fail with `IOError: Failed to find fully trained model in ...`.

A stage-2 run that dies partway must be restarted clean rather than resumed.
The transfer only fires when `start_epoch == 0`, so a resumed stage 2 silently
continues without the pretrained weights.

## Data layout

```
$BASE_DIR/data/crystal_dataset/
    train/real/    train/synth/
    val/real/      val/synth/
```

One tree serves both stages. Pretraining uses the `imagedir` dataset, which is
`torchvision.datasets.ImageFolder(path/train)` and therefore needs class
subdirectories — it sees `real` and `synth` as the two classes. Translation uses
`image-domain-hierarchy`, which reads `crystal_dataset/<split>/<domain>`
directly.

The train/val split is decided per **source photograph**, before any crop is
written. The real domain has three levels: one micrograph (the CSV's `raw_png`
column) yields several object crops (`name_PLB`, one per CSV row), and each
object crop yields `REAL_SAMPLES_PER_IMAGE = 5` sub-crops. The split key is
`raw_png`, the topmost level, so every object and every sub-crop from one
photograph lands in the same split.

Grouping only per object crop would not be enough. Two objects cropped out of
one micrograph are different regions of a single photograph — same specimen,
grid, session and defocus — so they are strongly correlated, and splitting them
apart leaks the validation set into training just as splitting the five
sub-crops of one object would. That specifically corrupts the memorisation
check: a val image whose near-duplicate was trained on looks memorised whether
or not the model memorised anything.

Output filenames are `{csv_row_index}_sample_{k}.png`, because multiple objects
share a `raw_png` and naming by photograph would collide. That means a
filename-based leak check on the generated tree — the sibling repository's
`scripts/fix_split.py` — can only verify that the five sub-crops of one object
stayed together. It cannot see which photograph a row came from, so it cannot
confirm photograph-level grouping; that guarantee comes from the generation
step, not from a check on the output.

## Note on the sibling repository

`diffusion-posterior-sampling` still references the old
`dataset/<domain>/<split>/` layout (for example in
`configs/crystal_cyclegan_config.yaml` and `scripts/train_diffusion.py`). It
needs to be updated separately to the new
`data/crystal_dataset/<split>/<domain>/` paths.
