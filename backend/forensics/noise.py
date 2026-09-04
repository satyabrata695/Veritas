"""
Stage 7 (physical/compression consistency, part 2): Noise-residual analysis.

Real camera sensors leave a fairly consistent noise "texture" across an
entire photo (shot noise + sensor pattern noise). Two things break that
consistency and are worth flagging:
  1. Splicing a region from a different source image (its local noise
     level won't match the surrounding area).
  2. Many generative pipelines over-smooth texture, producing unnaturally
     low and *uniform* noise almost everywhere.

We estimate a per-block noise level using a high-pass (Laplacian) residual
and look at both the overall level and the block-to-block variance.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def _laplacian_noise_map(gray: np.ndarray, block: int = 24) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = lap.shape
    rows = h // block
    cols = w // block
    noise_map = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            patch = lap[r * block:(r + 1) * block, c * block:(c + 1) * block]
            noise_map[r, c] = patch.std()
    return noise_map


def analyze_noise(image: Image.Image) -> dict[str, Any]:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    noise_map = _laplacian_noise_map(gray)

    mean_noise = float(noise_map.mean())
    std_noise = float(noise_map.std())
    coeff_of_var = float(std_noise / mean_noise) if mean_noise > 1e-6 else 0.0

    flags: list[str] = []

    # Heuristic thresholds tuned loosely; treat as a starting point to
    # calibrate against the labeled evaluation dataset in section 12.
    over_smooth_signal = 0.0
    if mean_noise < 1.5:
        over_smooth_signal = 0.6
        flags.append("Overall texture noise is unusually low/smooth across the whole image, a pattern often seen in synthetic images.")

    splice_signal = 0.0
    if coeff_of_var > 0.9:
        splice_signal = min(1.0, (coeff_of_var - 0.9) * 1.5)
        flags.append("Noise level varies sharply between regions, consistent with a spliced or composited area.")

    return {
        "mean_noise_level": round(mean_noise, 4),
        "noise_variability": round(coeff_of_var, 4),
        "synthetic_support_score": round(over_smooth_signal, 3),
        "splice_support_score": round(splice_signal, 3),
        "flags": flags,
    }
