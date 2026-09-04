"""Noise analysis — pure Python + Pillow only (no numpy/opencv)."""
from __future__ import annotations
import math
from typing import Any
from PIL import Image


def analyze_noise(image: Image.Image) -> dict[str, Any]:
    img = image.convert("L")
    w, h = img.size
    pixels = list(img.getdata())

    block = 24
    rows_b = h // block
    cols_b = w // block

    block_stds: list[float] = []
    for rb in range(rows_b):
        for cb in range(cols_b):
            patch: list[float] = []
            for r in range(rb * block, (rb + 1) * block):
                for c in range(cb * block, (cb + 1) * block):
                    # Laplacian at interior pixels only
                    if 0 < r < h - 1 and 0 < c < w - 1:
                        lap = (
                            pixels[(r - 1) * w + c] + pixels[(r + 1) * w + c] +
                            pixels[r * w + c - 1] + pixels[r * w + c + 1] -
                            4 * pixels[r * w + c]
                        )
                        patch.append(abs(lap))
            if patch:
                mean_p = sum(patch) / len(patch)
                var_p = sum((x - mean_p) ** 2 for x in patch) / len(patch)
                block_stds.append(math.sqrt(var_p))

    if not block_stds:
        return {
            "mean_noise_level": 0.0,
            "noise_variability": 0.0,
            "synthetic_support_score": 0.0,
            "splice_support_score": 0.0,
            "flags": [],
        }

    mean_noise = sum(block_stds) / len(block_stds)
    var_noise = sum((x - mean_noise) ** 2 for x in block_stds) / len(block_stds)
    std_noise = math.sqrt(var_noise)
    coeff_of_var = std_noise / mean_noise if mean_noise > 1e-6 else 0.0

    flags: list[str] = []
    over_smooth_signal = 0.0
    if mean_noise < 1.5:
        over_smooth_signal = 0.6
        flags.append("Overall texture noise is unusually low/smooth, a pattern often seen in synthetic images.")

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
