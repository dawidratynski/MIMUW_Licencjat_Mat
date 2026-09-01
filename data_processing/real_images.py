import os
import pandas as pd
import PIL.Image as pil
from PIL.Image import Image

from logging import warn, info, error, debug

from data_processing.config import *
from data_processing.sample import extract_sample
from data_processing.split import assign_splits, domain_dir


def process_real_images():
    info("Starting processing of real images")

    if not os.path.isdir(REAL_DATA_DIR):
        raise FileNotFoundError(
            f"Real image directory not found: {REAL_DATA_DIR}. It must hold the "
            f"JPEGs named in the 'name_PLB' column of {REAL_DATA_CSV_FILE}."
        )

    df = pd.read_csv(REAL_DATA_CSV_FILE)
    info(f"Loaded dataset CSV with {len(df)} rows")

    # Medatadata CSV header for reference:
    # name_PLB,img_number,raw_png,resolution_(px/nm),width,height,score

    sources = []
    checked = 0
    first_missing = None
    for ix, row in df.iterrows():
        if REAL_IMAGE_COUNT_LIMIT is not None and ix >= REAL_IMAGE_COUNT_LIMIT:
            break

        image_path = os.path.join(REAL_DATA_DIR, row["name_PLB"])
        checked += 1

        if not os.path.exists(image_path):
            warn(f"Missing image: {row['name_PLB']}")
            if first_missing is None:
                first_missing = row["name_PLB"]
            continue

        sources.append(
            (ix, image_path, float(row["resolution_(px/nm)"]), source_key(ix, row))
        )

    if not sources:
        raise FileNotFoundError(
            f"None of the {checked} images named in {REAL_DATA_CSV_FILE} were found "
            f"under {REAL_DATA_DIR}, starting with {first_missing!r}. Check that "
            f"REAL_DATA_DIR points at the directory holding the per-object JPEGs and "
            f"that their names match the CSV's 'name_PLB' column."
        )

    # Several rows are different objects cropped out of one micrograph, so the
    # split is decided per micrograph: objects from one photograph share the
    # specimen, grid, session and defocus and would leak across train/val.
    photographs = {key for _, _, _, key in sources}
    splits = assign_splits(photographs)

    info(
        f"Real domain split keyed on source photograph: {len(photographs)} "
        f"photographs / {len(sources)} objects / "
        f"{len(sources) * REAL_SAMPLES_PER_IMAGE} files"
    )

    written = 0
    for ix, image_path, resolution_px_per_nm, key in sources:
        try:
            process_image(
                ix, image_path, resolution_px_per_nm, domain_dir(splits[key], "real")
            )
            written += REAL_SAMPLES_PER_IMAGE
        except Exception as e:
            error(f"Error during processing image {image_path}: {e}")

    info(
        f"Finished processing real images successfully: {len(photographs)} "
        f"photographs / {len(sources)} objects / {written} files."
    )


def source_key(row_ix: int, row) -> str:
    raw_png = row.get("raw_png")

    if raw_png is None or pd.isna(raw_png) or not str(raw_png).strip():
        # No source photograph recorded: make the row its own group rather than
        # silently lumping every such row into one giant group.
        return f"__row_{row_ix}__"

    return str(raw_png).strip()


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
