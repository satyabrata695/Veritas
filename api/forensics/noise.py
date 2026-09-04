"""
Stage 7 (physical/compression consistency, part 2): Noise-residual analysis.
Rewritten to use only numpy + Pillow (no opencv) for Vercel compatibility.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def _laplacian_noise_map(gray: np.ndarray, block: int = 24) -> np.ndarray:
    # Approximate Laplacian using numpy convolution
    from numpy.lib.stride_tricks import as_strided
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # Pad and convolve manually (simple, no scipy needed)
    padded = np.pad(gray.astype(np.float64), 1, mode='reflect')
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] +
        padded[1:-1, :-2] + padded[1:-1, 2:] -
        4 * padded[1:-1, 1:-1]
    )
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
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    noise_map = _laplacian_noise_map(gray)

    mean_noise = float(noise_map.mean())
    std_noise = float(noise_map.std())
    coeff_of_var = float(std_noise / mean_noise) if mean_noise > 1e-6 else 0.0

    flags: list[str] = []

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
