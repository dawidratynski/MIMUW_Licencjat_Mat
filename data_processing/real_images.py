import os
import pandas as pd
import PIL.Image as pil
from PIL.Image import Image

from logging import warn, info, error, debug

from data_processing.config import *
from data_processing.sample import extract_sample


def process_real_images():
    info("Starting processing of real images")

    target_dir = os.path.join(OUTPUT_DIR, "real")
    os.makedirs(target_dir, exist_ok=True)

    df = pd.read_csv(REAL_DATA_CSV_FILE)
    info(f"Loaded dataset CSV with {len(df)} rows")

    # Medatadata CSV header for reference:
    # name_PLB,img_number,raw_png,resolution_(px/nm),width,height,score

    for ix, row in df.iterrows():
        if REAL_IMAGE_COUNT_LIMIT is not None and ix >= REAL_IMAGE_COUNT_LIMIT:
            break

        filename: str = row["name_PLB"]
        resolution_px_per_nm: float = float(row["resolution_(px/nm)"])

        image_path = os.path.join(REAL_DATA_DIR, filename)

        if not os.path.exists(image_path):
            warn(f"Missing image: {filename}")
            continue

        try:
            process_image(ix, image_path, resolution_px_per_nm, target_dir)
        except Exception as e:
            error(f"Error during processing image {image_path}: {e}")

    info("Finished processing real images successfully.")


def process_image(
    image_ix: str, filename: str, resolution_px_per_nm: float, target_dir: str
):
    debug(f"Processing image {filename}")

    with pil.open(filename) as img:
        for sample_ix in range(REAL_SAMPLES_PER_IMAGE):
            sample: Image = extract_sample(img, resolution_px_per_nm)
            output_filename = f"{image_ix}_sample_{sample_ix}.png"
            output_path = os.path.join(target_dir, output_filename)
            sample.save(output_path, "PNG")
