"""AI-generation detection — pure Python + Pillow only (no numpy/opencv)."""
from __future__ import annotations
import math
from typing import Any
from PIL import Image


def _gray_pixels(image: Image.Image, max_side: int = 512) -> list[list[float]]:
    """Convert to grayscale and return as 2D list, capped at max_side."""
    img = image.convert("L")
    w, h = img.size
    side = min(w, h, max_side)
    img = img.crop((0, 0, side, side))
    pixels = list(img.getdata())
    return [pixels[r * side:(r + 1) * side] for r in range(side)]


def _fft_magnitude_2d(gray: list[list[float]]) -> list[list[float]]:
    """Compute 2-D DFT magnitude via two passes of 1-D DFT (slow but correct)."""
    N = len(gray)
    M = len(gray[0]) if gray else 0

    def dft_1d(row: list[float]) -> list[complex]:
        n = len(row)
        return [
            sum(row[k] * math.e ** (-2j * math.pi * j * k / n) for k in range(n))
            for j in range(n)
        ]

    row_dft = [dft_1d(row) for row in gray]
    magnitude: list[list[float]] = []
    for j in range(M):
        col = [row_dft[i][j] for i in range(N)]
        col_dft = dft_1d(col)
        magnitude.append([math.log1p(abs(v)) for v in col_dft])
    return magnitude


def _radial_rings(magnitude: list[list[float]]) -> list[float]:
    """Bin magnitude by radius from centre."""
    rows = len(magnitude)
    cols = len(magnitude[0]) if magnitude else 0
    cy, cx = rows // 2, cols // 2
    max_r = min(cy, cx)
    ring_sum = [0.0] * max_r
    ring_cnt = [0] * max_r
    for r in range(rows):
        for c in range(cols):
            radius = int(math.sqrt((r - cy) ** 2 + (c - cx) ** 2))
            if radius < max_r:
                ring_sum[radius] += magnitude[r][c]
                ring_cnt[radius] += 1
    return [ring_sum[i] / ring_cnt[i] if ring_cnt[i] else 0.0 for i in range(max_r)]


def spectral_heuristic(image: Image.Image) -> dict[str, Any]:
    # Use a very small crop for speed in serverless env
    img = image.convert("L").crop((0, 0, min(image.width, 64), min(image.height, 64)))
    pixels = list(img.getdata())
    w, h = img.size
    gray = [pixels[r * w:(r + 1) * w] for r in range(h)]

    # Simplified spectral proxy: measure high-frequency content via Laplacian
    lap_vals: list[float] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            lap = (
                gray[r - 1][c] + gray[r + 1][c] +
                gray[r][c - 1] + gray[r][c + 1] -
                4 * gray[r][c]
            )
            lap_vals.append(abs(lap))

    if not lap_vals:
        mean_lap = 0.0
        std_lap = 0.0
    else:
        mean_lap = sum(lap_vals) / len(lap_vals)
        var = sum((x - mean_lap) ** 2 for x in lap_vals) / len(lap_vals)
        std_lap = math.sqrt(var)

    # Edge density via Sobel proxy
    edge_count = 0
    total = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            gx = gray[r - 1][c + 1] + 2 * gray[r][c + 1] + gray[r + 1][c + 1] \
               - gray[r - 1][c - 1] - 2 * gray[r][c - 1] - gray[r + 1][c - 1]
            gy = gray[r + 1][c - 1] + 2 * gray[r + 1][c] + gray[r + 1][c + 1] \
               - gray[r - 1][c - 1] - 2 * gray[r - 1][c] - gray[r - 1][c + 1]
            if math.sqrt(gx * gx + gy * gy) > 30:
                edge_count += 1
            total += 1

    edge_density = edge_count / total if total else 0.0

    signals: list[tuple[str, float]] = []
    flags: list[str] = []

    if mean_lap < 2.0 and std_lap < 3.0:
        signals.append(("uniform_low_texture", 0.25))
        flags.append("Texture/noise is unusually low and uniform.")

    if edge_density < 0.025:
        signals.append(("low_edge_density", 0.15))
        flags.append("The image has unusually low edge density.")

    raw_score = min(0.95, sum(w for _, w in signals))
    signal_count = len(signals)
    detector_confidence = min(0.85, 0.25 + 0.15 * signal_count) if signal_count else 0.10
    evidence_strength = "moderate" if signal_count >= 2 else ("weak" if signal_count == 1 else "none")

    return {
        "method": "laplacian_edge_heuristic_v1",
        "model_backed": False,
        "synthetic_probability": round(raw_score, 3),
        "detector_confidence": round(detector_confidence, 3),
        "evidence_strength": evidence_strength,
        "features": {
            "mean_texture_noise": round(mean_lap, 4),
            "texture_noise_std": round(std_lap, 4),
            "edge_density": round(edge_density, 4),
        },
        "flags": flags,
        "note": "Heuristic only. A low score does not establish authenticity.",
    }


class AIGenerationDetector:
    def __init__(self, model=None) -> None:
        self.model = model

    def run(self, image: Image.Image) -> dict[str, Any]:
        return spectral_heuristic(image)


AI_DETECTOR = AIGenerationDetector()