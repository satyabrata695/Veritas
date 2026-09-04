"""
Stage 7 (physical/compression consistency, part 1): Error Level Analysis.

Classic JPEG forensic technique: re-save the image at a known quality and
diff it against the original. Regions that were edited/composited after the
last save tend to sit at a different compression "error level" than the
rest of the image, showing up as a brighter patch in the ELA map.

Caveats we surface honestly in the report:
- Only meaningful for JPEG-family sources; PNG/WebP re-encodes differently.
- Heavy platform re-compression (e.g. repeated social-media re-uploads)
  can wash out or fake ELA signal in both directions.
- This is a *supporting* signal, not a standalone verdict.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageChops


def analyze_ela(image: Image.Image, original_format: str | None, quality: int = 90) -> dict[str, Any]:
    applicable = (original_format or "").upper() in {"JPEG", "JPG"}

    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(rgb, resaved)
    diff_arr = np.asarray(diff).astype(np.float32)

    # Per-pixel max across channels, then look at the distribution.
    intensity = diff_arr.max(axis=2)
    mean_error = float(intensity.mean())
    p95_error = float(np.percentile(intensity, 95))
    max_error = float(intensity.max())

    # Split into a coarse grid and flag blocks whose error sits far above
    # the image-wide baseline - a simple stand-in for "localized editing".
    h, w = intensity.shape
    block = 32
    hotspots = 0
    total_blocks = 0
    baseline = mean_error + 1e-6
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            total_blocks += 1
            block_mean = intensity[y:y + block, x:x + block].mean()
            if block_mean > baseline * 4:
                hotspots += 1
    hotspot_ratio = (hotspots / total_blocks) if total_blocks else 0.0

    # Rough manipulation-supporting score. Only trusted when `applicable`.
    raw_score = min(1.0, (hotspot_ratio * 3) + (p95_error / 255) * 0.5)
    score = round(raw_score, 3) if applicable else round(raw_score * 0.3, 3)

    flags = []
    if not applicable:
        flags.append(f"Source format is {original_format or 'unknown'}; ELA is most reliable on JPEG and is down-weighted here.")
    if applicable and hotspot_ratio > 0.03:
        flags.append(f"Localized high-error regions detected in {hotspots}/{total_blocks} blocks - possible localized edit.")

    return {
        "applicable": applicable,
        "mean_error": round(mean_error, 3),
        "p95_error": round(p95_error, 3),
        "max_error": round(max_error, 3),
        "hotspot_block_ratio": round(hotspot_ratio, 3),
        "manipulation_support_score": score,
        "flags": flags,
    }
