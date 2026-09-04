"""
Stage 6: Deepfake / face-manipulation analysis.

Pipeline: detect faces (Haar cascade - fast, dependency-light, ships with
OpenCV) -> crop/align -> run manipulation heuristics on each face crop.

The heuristics here (edge-density texture check, left/right symmetry
check, boundary-blur check around the face contour) are classic, cheap
signals used as an MVP placeholder. For a competitive hackathon result,
swap `_heuristic_manipulation_score()` for a trained face-forensics model
(e.g. a fine-tuned Xception/EfficientNet on FaceForensics++-style data) -
the surrounding face-detection/crop/report plumbing does not need to change.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _heuristic_manipulation_score(face_bgr: np.ndarray) -> tuple[float, list[str]]:
    flags: list[str] = []
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Texture: manipulated/blended faces often show locally over-smooth
    # patches (e.g. around a swapped mouth/eye region) next to sharper skin.
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(edges.mean() / 255)

    # 2. Symmetry: real faces are close-to-but-not-perfectly symmetric;
    # crude face swaps sometimes over- or under-shoot natural symmetry.
    flipped = cv2.flip(gray, 1)
    if flipped.shape == gray.shape:
        diff = cv2.absdiff(gray, flipped).astype(np.float32)
        asymmetry = float(diff.mean() / 255)
    else:
        asymmetry = 0.2  # neutral fallback

    # 3. Boundary blur ring: compare sharpness just inside vs. just outside
    # the detected face box - blending artifacts often soften this ring.
    border = max(2, w // 20)
    inner = gray[border:-border, border:-border] if h > 2 * border and w > 2 * border else gray
    inner_lap_var = float(cv2.Laplacian(inner, cv2.CV_64F).var())
    outer_lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ring_ratio = inner_lap_var / (outer_lap_var + 1e-6)

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
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

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
        crop = bgr[y:y + h, x:x + w]
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
