import random
import math
import PIL.Image as pil
from PIL.Image import Image


from data_processing.config import *


def get_sample_angle(sample_size: int, max_dim: int) -> tuple[float, int]:
    for _ in range(100):
        angle = random.uniform(0, 360)
        theta = math.radians(angle)
        required_margin = math.ceil(
            sample_size * (abs(math.cos(theta)) + abs(math.sin(theta))) / 2
        )
        if 2 * required_margin <= max_dim:
            return angle, required_margin

    # fallback to no rotation
    return 0.0, sample_size // 2


def sample_center(min_c, max_c, width, strength=0.7) -> tuple[int, int]:
    cx = int(random.gauss(width / 2, (max_c - min_c) * CENTER_BIAS_STD))
    cx = max(min_c, min(cx, max_c))
    return cx


def extract_sample(
    image: Image,
    resolution_px_per_nm: float,
) -> Image:
    width, height = image.size

    # Sample size in pixels before resizing to target px/nm resolution
    sample_size_px = math.ceil(SAMPLE_SIZE_NM * resolution_px_per_nm)

    if sample_size_px > min(width, height):
        raise ValueError(f"Image too small to extract samples")

    angle, sample_margin = get_sample_angle(sample_size_px, min(width, height))

    # --- Valid center bounds ---
    min_cx = sample_margin
    max_cx = width - sample_margin
    min_cy = sample_margin
    max_cy = height - sample_margin

    # --- Biased random center ---
    cx = sample_center(min_cx, max_cx, width, CENTER_BIAS_STD)
    cy = sample_center(min_cy, max_cy, width, CENTER_BIAS_STD)

    sample = (
        image.convert("RGB")
        .crop(
            (
                cx - sample_margin,
                cy - sample_margin,
                cx + sample_margin,
                cy + sample_margin,
            )
        )
        .rotate(angle, resample=pil.Resampling.BICUBIC, expand=False)
        .crop(
            (
                sample_margin - math.floor(sample_size_px / 2),
                sample_margin - math.floor(sample_size_px / 2),
                sample_margin + math.ceil(sample_size_px / 2),
                sample_margin + math.ceil(sample_size_px / 2),
            )
        )
        .resize((SAMPLE_SIZE_PX, SAMPLE_SIZE_PX), pil.Resampling.LANCZOS)
    )

    if TRANSPOSE_SAMPLES and random.random() < 0.5:
        sample = sample.transpose(pil.Transpose.FLIP_LEFT_RIGHT)

    return sample
