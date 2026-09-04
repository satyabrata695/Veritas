"""AI-generation detection for Veritas.

Uses the trained ViT classifier when the model is available.
Falls back to the existing conservative heuristic if the trained
model cannot be loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image


# ============================================================
# OPTIONAL TRAINED MODEL IMPORTS
# ============================================================

try:
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
    )

    TRAINED_MODEL_AVAILABLE = True
except ImportError:
    TRAINED_MODEL_AVAILABLE = False


# ============================================================
# PATH TO TRAINED VERITAS MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINED_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "ai_detector"
)


# ============================================================
# OLD HEURISTIC
# ============================================================

def spectral_heuristic(image: Image.Image) -> dict[str, Any]:
    """Conservative fallback detector.

    This is only used when the trained model is unavailable.
    A low score must never be interpreted as proof of authenticity.
    """

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    h, w = rgb.shape[:2]

    side = min(h, w, 512)

    rgb = rgb[:side, :side]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Spectral analysis
    # --------------------------------------------------------

    spectrum = np.fft.fftshift(
        np.fft.fft2(gray)
    )

    magnitude = np.log1p(
        np.abs(spectrum)
    )

    cy = magnitude.shape[0] // 2
    cx = magnitude.shape[1] // 2

    yy, xx = np.mgrid[
        0:magnitude.shape[0],
        0:magnitude.shape[1],
    ]

    radius_map = np.sqrt(
        (yy - cy) ** 2 +
        (xx - cx) ** 2
    ).astype(np.int32)

    max_radius = min(cy, cx)

    ring_sum = np.bincount(
        radius_map.ravel(),
        magnitude.ravel(),
    )

    ring_count = np.bincount(
        radius_map.ravel()
    )

    rings = ring_sum / np.maximum(
        ring_count,
        1,
    )

    rings = (
        rings[:max_radius]
        if max_radius > 0
        else rings
    )

    if len(rings) > 16:

        q1 = rings[
            : len(rings) // 4
        ].mean()

        q3 = rings[
            3 * len(rings) // 4 :
        ].mean()

        rolloff = float(
            q3 / (q1 + 1e-6)
        )

        deriv = np.abs(
            np.diff(rings)
        )

        spike_ratio = float(
            (
                deriv
                >
                (
                    deriv.mean()
                    +
                    3.0 * deriv.std()
                    +
                    1e-6
                )
            ).mean()
        )

    else:
        rolloff = 1.0
        spike_ratio = 0.0

    # --------------------------------------------------------
    # Texture/noise
    # --------------------------------------------------------

    lap = cv2.Laplacian(
        gray.astype(np.float32),
        cv2.CV_32F,
    )

    block = 32
    levels: list[float] = []

    for y in range(
        0,
        gray.shape[0] - block + 1,
        block,
    ):
        for x in range(
            0,
            gray.shape[1] - block + 1,
            block,
        ):

            patch = lap[
                y:y + block,
                x:x + block,
            ]

            levels.append(
                float(patch.std())
            )

    if levels:

        noise_mean = float(
            np.mean(levels)
        )

        noise_cv = float(
            np.std(levels)
            /
            (noise_mean + 1e-6)
        )

    else:

        noise_mean = float(
            lap.std()
        )

        noise_cv = 0.0

    # --------------------------------------------------------
    # Edge structure
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray.astype(np.uint8),
        60,
        150,
    )

    edge_density = float(
        edges.mean() / 255.0
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    signals: list[tuple[str, float]] = []
    flags: list[str] = []

    if rolloff < 0.28:

        signals.append(
            (
                "rapid_spectral_rolloff",
                0.30,
            )
        )

        flags.append(
            "High-frequency spectrum falls off unusually quickly."
        )

    if spike_ratio > 0.07:

        signals.append(
            (
                "spectral_spikes",
                0.25,
            )
        )

        flags.append(
            "Periodic spectral spikes were detected."
        )

    if (
        noise_mean < 2.0
        and noise_cv < 0.25
    ):

        signals.append(
            (
                "uniform_low_texture",
                0.25,
            )
        )

        flags.append(
            "Texture/noise is unusually low and uniform."
        )

    if (
        edge_density < 0.025
        and gray.size > 10000
    ):

        signals.append(
            (
                "low_edge_density",
                0.15,
            )
        )

        flags.append(
            "The image has unusually low edge density."
        )

    raw_score = min(
        0.95,
        sum(weight for _, weight in signals),
    )

    signal_count = len(signals)

    detector_confidence = (
        min(
            0.85,
            0.25 + 0.15 * signal_count,
        )
        if signal_count
        else 0.10
    )

    if signal_count >= 2:
        evidence_strength = "moderate"
    elif signal_count == 1:
        evidence_strength = "weak"
    else:
        evidence_strength = "none"

    return {
        "method": "multi_cue_spectral_heuristic_v1",
        "model_backed": False,
        "synthetic_probability": round(
            raw_score,
            3,
        ),
        "detector_confidence": round(
            detector_confidence,
            3,
        ),
        "evidence_strength": evidence_strength,
        "features": {
            "high_freq_rolloff": round(
                rolloff,
                4,
            ),
            "spectral_spike_ratio": round(
                spike_ratio,
                4,
            ),
            "mean_texture_noise": round(
                noise_mean,
                4,
            ),
            "texture_noise_cv": round(
                noise_cv,
                4,
            ),
            "edge_density": round(
                edge_density,
                4,
            ),
        },
        "flags": flags,
        "note": (
            "Heuristic only. "
            "A low score does not establish authenticity."
        ),
    }


# ============================================================
# TRAINED MODEL INTERFACE
# ============================================================

class TrainedClassifier(Protocol):

    def predict_proba(
        self,
        image: Image.Image,
    ) -> float:
        ...


class ViTClassifier:
    """Loads the trained Veritas ViT model."""

    def __init__(
        self,
        model_dir: Path,
    ) -> None:

        if not TRAINED_MODEL_AVAILABLE:
            raise RuntimeError(
                "PyTorch and/or Transformers are not installed."
            )

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Trained model directory not found: {model_dir}"
            )

        if not (
            model_dir / "config.json"
        ).exists():

            raise FileNotFoundError(
                f"config.json not found in {model_dir}"
            )

        print(
            f"[Veritas] Loading trained AI detector from: "
            f"{model_dir}"
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = (
            AutoImageProcessor.from_pretrained(
                str(model_dir)
            )
        )

        self.model = (
            AutoModelForImageClassification.from_pretrained(
                str(model_dir)
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.ai_label_id = self._find_ai_label_id()

        print(
            f"[Veritas] AI detector loaded on "
            f"{self.device}"
        )

        print(
            f"[Veritas] AI label ID = "
            f"{self.ai_label_id}"
        )

    def _find_ai_label_id(self) -> int:
        """Find the class ID corresponding to AI-generated."""

        label_mapping = (
            self.model.config.id2label
        )

        for raw_id, label in label_mapping.items():

            label_text = str(
                label
            ).strip().lower()

            if (
                "ai" in label_text
                or
                "generated" in label_text
                or
                "synthetic" in label_text
            ):

                return int(raw_id)

        raise RuntimeError(
            "Could not find the AI-generated class in model labels. "
            f"Found labels: {label_mapping}"
        )

    def predict_proba(
        self,
        image: Image.Image,
    ) -> float:

        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        return float(
            probabilities[
                self.ai_label_id
            ].item()
        )


# ============================================================
# AI GENERATION DETECTOR
# ============================================================

class AIGenerationDetector:
    """Uses trained ViT when available, otherwise heuristic."""

    def __init__(
        self,
        model: TrainedClassifier | None = None,
    ) -> None:

        self.model = model

    def run(
        self,
        image: Image.Image,
    ) -> dict[str, Any]:

        if self.model is None:
            return spectral_heuristic(image)

        proba = max(
            0.0,
            min(
                1.0,
                float(
                    self.model.predict_proba(
                        image
                    )
                ),
            ),
        )

        return {
            "method": "veritas_vit_ai_detector",
            "model_backed": True,
            "synthetic_probability": round(
                proba,
                4,
            ),
            "detector_confidence": round(
                max(
                    proba,
                    1.0 - proba,
                ),
                4,
            ),
            "evidence_strength": "strong",
            "flags": [],
            "note": (
                "Probability produced by the trained "
                "Veritas ViT classifier."
            ),
        }


# ============================================================
# LOAD THE TRAINED MODEL ONCE
# ============================================================

TRAINED_CLASSIFIER: ViTClassifier | None = None

if TRAINED_MODEL_AVAILABLE:

    try:

        TRAINED_CLASSIFIER = ViTClassifier(
            TRAINED_MODEL_DIR
        )

    except Exception as exc:

        print(
            "[Veritas] WARNING: Could not load trained "
            "AI detector."
        )

        print(
            f"[Veritas] Reason: {exc}"
        )

        print(
            "[Veritas] Falling back to heuristic detector."
        )


# Global detector used by the backend.
AI_DETECTOR = AIGenerationDetector(
    model=TRAINED_CLASSIFIER
)