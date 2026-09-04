"""
Stage 3: Metadata analysis.

Reads EXIF data (when present) and flags patterns that are *supporting*
evidence only. Per the design principle in the proposal: missing metadata
must NOT be treated as proof of forgery, since it is trivially stripped by
messaging apps, social platforms, and screenshots.
"""
from __future__ import annotations

from typing import Any
from PIL import Image
from PIL.ExifTags import TAGS

# Software tags commonly left behind by editors / generators. This is a
# starting heuristic list for the hackathon MVP - extend with a maintained
# list or a small classifier as the project matures.
EDITING_SOFTWARE_HINTS = [
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
    "canva", "snapseed", "facetune",
]
GENERATIVE_SOFTWARE_HINTS = [
    "midjourney", "stable diffusion", "dall", "dalle", "dall-e",
    "firefly", "runway", "leonardo.ai", "ideogram", "flux",
]


def analyze_metadata(image: Image.Image) -> dict[str, Any]:
    raw_exif = image.getexif()
    exif: dict[str, Any] = {}
    for tag_id, value in raw_exif.items():
        tag = TAGS.get(tag_id, tag_id)
        # Keep values JSON-serializable.
        if isinstance(value, (bytes, bytearray)):
            continue
        exif[str(tag)] = str(value)

    software = str(exif.get("Software", "")).lower()
    make = exif.get("Make")
    model = exif.get("Model")
    datetime_original = exif.get("DateTimeOriginal") or exif.get("DateTime")

    flags: list[str] = []
    generative_hit = next((h for h in GENERATIVE_SOFTWARE_HINTS if h in software), None)
    editing_hit = next((h for h in EDITING_SOFTWARE_HINTS if h in software), None)

    if generative_hit:
        flags.append(f"Software tag references a generative tool ('{generative_hit}').")
    if editing_hit:
        flags.append(f"Software tag references an editing tool ('{editing_hit}').")
    if not exif:
        flags.append("No EXIF metadata present (common after re-saving, screenshots, or platform re-upload).")
    if exif and not make and not model:
        flags.append("EXIF present but camera make/model fields are missing.")

    # A very rough anomaly score - NOT a verdict on its own. Weighted lightly
    # in fusion because metadata is easy to strip or forge in either direction.
    score = 0.0
    if generative_hit:
        score = 0.85
    elif editing_hit:
        score = 0.5
    elif not exif:
        score = 0.35
    elif not make and not model:
        score = 0.3

    return {
        "has_exif": bool(exif),
        "camera_make": make,
        "camera_model": model,
        "software_tag": exif.get("Software"),
        "datetime_original": datetime_original,
        "flags": flags,
        "anomaly_score": round(score, 3),
        "raw_exif_keys": list(exif.keys()),
    }
