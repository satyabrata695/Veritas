"""ELA (Error Level Analysis) — pure Python + Pillow only (no numpy)."""
from __future__ import annotations
import io
import math
from typing import Any
from PIL import Image, ImageChops


def analyze_ela(image: Image.Image, original_format: str | None, quality: int = 90) -> dict[str, Any]:
    applicable = (original_format or "").upper() in {"JPEG", "JPG"}

    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(rgb, resaved)
    pixels = list(diff.getdata())  # list of (R,G,B) tuples

    # Per-pixel max across channels
    intensity = [max(p) for p in pixels]
    W, H = diff.size
    n = len(intensity)

    mean_error = sum(intensity) / n if n else 0.0
    sorted_i = sorted(intensity)
    p95_error = sorted_i[int(0.95 * n)] if n else 0.0
    max_error = sorted_i[-1] if sorted_i else 0.0

    # Block hotspot detection
    block = 32
    hotspots = 0
    total_blocks = 0
    baseline = mean_error + 1e-6

    for y in range(0, H - block, block):
        for x in range(0, W - block, block):
            total_blocks += 1
            block_sum = 0.0
            count = 0
            for r in range(y, y + block):
                for c in range(x, x + block):
                    block_sum += intensity[r * W + c]
                    count += 1
            block_mean = block_sum / count if count else 0.0
            if block_mean > baseline * 4:
                hotspots += 1

    hotspot_ratio = (hotspots / total_blocks) if total_blocks else 0.0
    raw_score = min(1.0, (hotspot_ratio * 3) + (p95_error / 255) * 0.5)
    score = round(raw_score, 3) if applicable else round(raw_score * 0.3, 3)

    flags = []
    if not applicable:
        flags.append(f"Source format is {original_format or 'unknown'}; ELA is most reliable on JPEG.")
    if applicable and hotspot_ratio > 0.03:
        flags.append(f"Localized high-error regions detected in {hotspots}/{total_blocks} blocks.")

    return {
        "applicable": applicable,
        "mean_error": round(mean_error, 3),
        "p95_error": round(p95_error, 3),
        "max_error": round(max_error, 3),
        "hotspot_block_ratio": round(hotspot_ratio, 3),
        "manipulation_support_score": score,
        "flags": flags,
    }
