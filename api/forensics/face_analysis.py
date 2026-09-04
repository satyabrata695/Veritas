"""Face analysis — pure Python + Pillow only (no numpy/opencv)."""
from __future__ import annotations
import math
from typing import Any
from PIL import Image


def _sobel_edge_density(pixels: list[int], w: int, h: int) -> float:
    edge_count = 0
    total = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            gx = (pixels[(r-1)*w+(c+1)] + 2*pixels[r*w+(c+1)] + pixels[(r+1)*w+(c+1)]
                - pixels[(r-1)*w+(c-1)] - 2*pixels[r*w+(c-1)] - pixels[(r+1)*w+(c-1)])
            gy = (pixels[(r+1)*w+(c-1)] + 2*pixels[(r+1)*w+c] + pixels[(r+1)*w+(c+1)]
                - pixels[(r-1)*w+(c-1)] - 2*pixels[(r-1)*w+c] - pixels[(r-1)*w+(c+1)])
            if math.sqrt(gx*gx + gy*gy) > 30:
                edge_count += 1
            total += 1
    return edge_count / total if total else 0.0


def _laplacian_var(pixels: list[int], w: int, h: int) -> float:
    laps: list[float] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            lap = float(
                pixels[(r-1)*w+c] + pixels[(r+1)*w+c] +
                pixels[r*w+(c-1)] + pixels[r*w+(c+1)] -
                4 * pixels[r*w+c]
            )
            laps.append(lap)
    if not laps:
        return 0.0
    mean = sum(laps) / len(laps)
    return sum((x - mean) ** 2 for x in laps) / len(laps)


def _detect_skin_region(image: Image.Image):
    """Skin-tone detection, returns (x,y,w,h) or None."""
    rgb = image.convert("RGB")
    W, H = rgb.size
    pixels = list(rgb.getdata())
    min_c, max_c, min_r, max_r = W, 0, H, 0
    found = False
    for r in range(H):
        for c in range(W):
            ri, gi, bi = pixels[r * W + c]
            if (ri > 60 and gi > 40 and bi > 20 and
                    ri > gi and ri > bi and
                    ri - gi > 15 and abs(int(ri) - int(bi)) > 15):
                found = True
                min_c = min(min_c, c)
                max_c = max(max_c, c)
                min_r = min(min_r, r)
                max_r = max(max_r, r)
    if not found:
        return None
    bw, bh = max_c - min_c, max_r - min_r
    if bw < 30 or bh < 30:
        return None
    return (min_c, min_r, bw, bh)


def analyze_faces(image: Image.Image) -> dict[str, Any]:
    # Use a small resize to keep serverless execution fast
    thumb = image.copy()
    thumb.thumbnail((128, 128))
    region = _detect_skin_region(thumb)

    if region is None:
        return {
            "faces_detected": 0,
            "max_manipulation_probability": 0.0,
            "faces": [],
            "flags": ["No faces detected - face-manipulation checks skipped."],
        }

    x, y, bw, bh = region
    crop = thumb.convert("L").crop((x, y, x + bw, y + bh))
    cw, ch = crop.size
    pixels = list(crop.getdata())

    score = 0.0
    flags: list[str] = []

    edge_density = _sobel_edge_density(pixels, cw, ch)
    if edge_density < 0.03:
        score += 0.25
        flags.append("Face region shows unusually low edge texture (possible over-smoothing).")

    # Asymmetry check
    rows = [pixels[r * cw:(r + 1) * cw] for r in range(ch)]
    diffs = [abs(rows[r][c] - rows[r][cw - 1 - c]) for r in range(ch) for c in range(cw // 2)]
    asymmetry = (sum(diffs) / len(diffs) / 255) if diffs else 0.2
    if asymmetry < 0.03 or asymmetry > 0.18:
        score += 0.25
        flags.append("Face symmetry is outside the typical natural range.")

    # Sharpness ring check
    border = max(2, cw // 20)
    if ch > 2 * border and cw > 2 * border:
        inner_pixels = []
        for r in range(border, ch - border):
            inner_pixels.extend(pixels[r * cw + border: r * cw + cw - border])
        inner_var = _laplacian_var(pixels, cw, ch)
        outer_var = _laplacian_var(pixels, cw, ch)
        ring_ratio = inner_var / (outer_var + 1e-6)
        if ring_ratio < 0.6:
            score += 0.3
            flags.append("Sharpness drops at the face boundary, consistent with a composited edit.")

    face_report = {
        "box": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
        "manipulation_probability": round(min(1.0, score), 3),
        "flags": flags,
    }
    return {
        "faces_detected": 1,
        "max_manipulation_probability": round(min(1.0, score), 3),
        "faces": [face_report],
        "flags": flags,
    }
