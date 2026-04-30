import os
import numpy as np
import PIL.Image

from PIL.Image import Image

from logging import warn, info, debug

from data_processing.config import *
from data_processing.sample import extract_sample


def process_synth_images() -> None:
    info("Starting processing of synth images")

    target_dir = os.path.join(OUTPUT_DIR, "synth")
    os.makedirs(target_dir, exist_ok=True)

    images_left_to_process = SYNTH_IMAGE_COUNT_LIMIT

    for filename in SYNTH_DATA_FILES:
        if images_left_to_process <= 0:
            break

        info(f"Processing file: {filename}")

        image_path = os.path.join(SYNTH_DATA_DIR, filename)

        images: np.ndarray = np.load(image_path, mmap_mode="r")
        if images.ndim != 3:
            raise ValueError(f"Unsupported shape: {images.shape}")

        for image_ix, image_raw in enumerate(images):
            if images_left_to_process <= 0:
                break

            # Ensure correct data type
            if image_raw.dtype != np.uint8:
                warn(
                    f"{filename} @ {image_ix}: Image has wrong dtype: {image_raw.dtype}. Converting to uint8."
                )
                image_raw = np.clip(image_raw, 0, 255).astype(np.uint8)

            image = PIL.Image.fromarray(image_raw)

            debug(f"Processing image {filename} @ {image_ix}")

            for sample_ix in range(SYNTH_SAMPLES_PER_IMAGE):
                sample: Image = extract_sample(image, SYNTH_IMAGE_RESOLUTION_PX_PER_NM)
                output_filename = f"{filename}_@_{image_ix}_sample_{sample_ix}.png"
                output_path = os.path.join(target_dir, output_filename)
                sample.save(output_path, "PNG")

            images_left_to_process -= 1

    info("Finished processing synth images successfully.")
