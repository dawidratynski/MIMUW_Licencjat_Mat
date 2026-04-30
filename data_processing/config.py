import os

# DATASET SETTINGS

BASE_DIR: str = "/mnt/e/Pobrane HDD"  # SET THIS TO DATA LOCATION ON YOUR MACHINE

OUTPUT_DIR: str = os.path.join(BASE_DIR, "dataset")

REAL_DATA_DIR: str = os.path.join(BASE_DIR, "raw/real")
REAL_DATA_CSV_FILE: str = os.path.join(BASE_DIR, "raw/data_summary_2957.csv")

SYNTH_DATA_DIR: str = os.path.join(BASE_DIR, "raw/synth")
SYNTH_DATA_FILES: list[str] = [
    "images_0-010.npy",
    "images_1-006.npy",
    "images_2-003.npy",
    "images_3-004.npy",
    "images_4-001.npy",
    "images_5-002.npy",
    "images_6-008.npy",
    "images_7-009.npy",
    "images_8-005.npy",
    "images_9-007.npy",
]


REMOVE_OUTPUT_DIR_IF_PRESENT = True
TRAIN_VAL_SPLIT_RATIO = 0.8

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
