import os

# DATASET SETTINGS

BASE_DIR: str = r"C:\crystal_data"  # SET THIS TO THE DATASET OUTPUT LOCATION

# Raw inputs are read from RAW_DIR, which is independent of BASE_DIR: the
# generated dataset is tens of thousands of files and must stay off the
# Dropbox-synced repository tree, while the raw inputs live inside it.
RAW_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "daneProsem"
)

# UVCGAN resolves {UVCGAN2_DATA}/{path}/..., where `path` is 'crystal_dataset',
# so UVCGAN2_DATA must point at UVCGAN_DATA_DIR.
UVCGAN_DATA_DIR: str = os.path.join(BASE_DIR, "data")
OUTPUT_DIR: str = os.path.join(UVCGAN_DATA_DIR, "crystal_dataset")

# The per-object crop directory: one JPEG per CSV row, named exactly as in the
# 'name_PLB' column (e.g. '123_oat_ETIO_twoweeks_30kx_(1)_obj2.jpg').
REAL_DATA_DIR: str = os.path.join(RAW_DIR, "crop_filtering_by_region")
REAL_DATA_CSV_FILE: str = os.path.join(RAW_DIR, "data_summary_2957.csv")

# The .npy stacks sit at the raw root alongside 'Kopia metadata.json', not in a
# 'synth' subdirectory.
SYNTH_DATA_DIR: str = RAW_DIR

# Empty means "discover 3-D *.npy image stacks in SYNTH_DATA_DIR, sorted".
# Companion params_*.npy files (and any other non-3-D .npy) are skipped.
# Listing the stacks explicitly is stricter: a listed stack that is missing
# or has the wrong ndim aborts the run.
SYNTH_DATA_FILES: list[str] = []


TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
SPLITS = (TRAIN_SPLIT, VAL_SPLIT)
DOMAINS = ("real", "synth")

REMOVE_OUTPUT_DIR_IF_PRESENT = True
TRAIN_VAL_SPLIT_RATIO = 0.8
TRAIN_VAL_SPLIT_SEED = 42

# SAMPLING SETTINGS
SAMPLE_SIZE_PX = 150
SAMPLE_SIZE_NM = 300
CENTER_BIAS_STD = 0.15
TRANSPOSE_SAMPLES = True

REAL_SAMPLES_PER_IMAGE = 5
REAL_IMAGE_COUNT_LIMIT = 2957

SYNTH_SAMPLES_PER_IMAGE = 1
SYNTH_IMAGE_COUNT_LIMIT = 15000
SYNTH_IMAGE_RESOLUTION_PX_PER_NM = 0.5
