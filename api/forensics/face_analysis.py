"""
Stage 6: Deepfake / face-manipulation analysis.
Rewritten to use only numpy + Pillow (no opencv) for Vercel compatibility.
Face detection uses a simple skin-tone region approach as a lightweight proxy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Compute edge magnitude via Sobel using pure numpy."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    g = gray.astype(np.float32)
    pad = np.pad(g, 1, mode='reflect')
    Gx = (
        -pad[:-2, :-2] + pad[:-2, 2:] +
        -2 * pad[1:-1, :-2] + 2 * pad[1:-1, 2:] +
        -pad[2:, :-2] + pad[2:, 2:]
    )
    Gy = (
        -pad[:-2, :-2] - 2 * pad[:-2, 1:-1] - pad[:-2, 2:] +
        pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:]
    )
    return np.sqrt(Gx ** 2 + Gy ** 2)


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of Laplacian — sharpness measure."""
    pad = np.pad(gray.astype(np.float64), 1, mode='reflect')
    lap = (
        pad[:-2, 1:-1] + pad[2:, 1:-1] +
        pad[1:-1, :-2] + pad[1:-1, 2:] -
        4 * pad[1:-1, 1:-1]
    )
    return float(lap.var())


def _detect_face_regions(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Lightweight skin-tone proxy for face detection.
    Returns list of (x, y, w, h) bounding boxes.
    No opencv required.
    """
    r, g, b = rgb[:, :, 0].astype(float), rgb[:, :, 1].astype(float), rgb[:, :, 2].astype(float)
    # Skin heuristic: warm, mid-range pixels
    skin = (
        (r > 60) & (g > 40) & (b > 20) &
        (r > g) & (r > b) &
        (r - g > 15) &
        (np.abs(r.astype(int) - b.astype(int)) > 15)
    )
    if skin.sum() < 500:
        return []

    rows = np.where(skin.any(axis=1))[0]
    cols = np.where(skin.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return []

    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    h, w = y1 - y0, x1 - x0

    if h < 30 or w < 30:
        return []

    return [(x0, y0, w, h)]


def _heuristic_manipulation_score(face_rgb: np.ndarray) -> tuple[float, list[str]]:
    flags: list[str] = []
    gray = np.mean(face_rgb, axis=2).astype(np.uint8)
    h, w = gray.shape

    edges = _sobel_edges(gray)
    edge_density = float((edges > 30).mean())

    flipped = gray[:, ::-1]
    asymmetry = float(np.abs(gray.astype(float) - flipped).mean() / 255)

    border = max(2, w // 20)
    inner = gray[border:-border, border:-border] if h > 2 * border and w > 2 * border else gray
    inner_sharp = _laplacian_var(inner.astype(np.float64))
    outer_sharp = _laplacian_var(gray.astype(np.float64))
    ring_ratio = inner_sharp / (outer_sharp + 1e-6)

    score = 0.0
    if edge_density < 0.03:
        score += 0.25
        flags.append("Face region shows unusually low edge texture (possible over-smoothing).")
    if asymmetry < 0.03 or asymmetry > 0.18:
        score += 0.25
        flags.append("Face symmetry is outside the typical natural range.")
    if ring_ratio < 0.6:
        score += 0.3
        flags.append("Sharpness drops noticeably at the face boundary, consistent with a blended/composited edit.")

    return round(min(1.0, score), 3), flags


def analyze_faces(image: Image.Image) -> dict[str, Any]:
    rgb = np.asarray(image.convert("RGB"))
    faces = _detect_face_regions(rgb)

    if len(faces) == 0:
        return {
            "faces_detected": 0,
            "max_manipulation_probability": 0.0,
            "faces": [],
            "flags": ["No faces detected - face-manipulation checks skipped."],
        }

    face_reports = []
    max_score = 0.0
    for (x, y, w, h) in faces:
        crop = rgb[y:y + h, x:x + w]
        score, flags = _heuristic_manipulation_score(crop)
        max_score = max(max_score, score)
        face_reports.append({
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "manipulation_probability": score,
            "flags": flags,
        })

    all_flags = [f for face in face_reports for f in face["flags"]]
    return {
        "faces_detected": len(faces),
        "max_manipulation_probability": max_score,
        "faces": face_reports,
        "flags": all_flags,
    }
