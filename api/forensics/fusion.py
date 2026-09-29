"""
Stage 8: Evidence fusion.

Combines independent signals into one verdict. Deliberately NOT a simple
average (section 9 of the proposal warns against that): unrelated scores
get different weights, contradictions reduce confidence instead of being
hidden, and there is an explicit "inconclusive" outcome when evidence is
weak or conflicting.

Calibrate the weights/thresholds below against the labeled evaluation
dataset described in section 12 before trusting this for anything beyond
a demo - these numbers are reasonable starting points, not measured values.
"""
from __future__ import annotations

from typing import Any

VERDICT_AUTHENTIC = "LIKELY AUTHENTIC"
VERDICT_AI_GENERATED = "LIKELY AI-GENERATED"
VERDICT_MANIPULATED = "LIKELY MANIPULATED"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


def fuse_evidence(
    metadata: dict[str, Any],
    provenance: dict[str, Any],
    ai_detection: dict[str, Any],
    face_analysis: dict[str, Any],
    ela: dict[str, Any],
    noise: dict[str, Any],
) -> dict[str, Any]:
    """Fuse evidence without treating an absence of red flags as authenticity.

    ``LIKELY AUTHENTIC`` requires positive evidence: either a verified C2PA
    manifest or a high-confidence real prediction from the trained classifier.
    Heuristic-only results with weak evidence are intentionally inconclusive.
    """
    detector_synthetic_prob = float(ai_detection.get("synthetic_probability", 0.0))
    synthetic_prob = 0.7 * detector_synthetic_prob + 0.3 * noise.get("synthetic_support_score", 0.0)

    manipulation_prob = max(
        face_analysis.get("max_manipulation_probability", 0.0),
        ela.get("manipulation_support_score", 0.0),
        noise.get("splice_support_score", 0.0),
    )

    metadata_anomaly = metadata.get("anomaly_score", 0.0)

    # Blend in a light metadata contribution without letting it dominate,
    # since metadata is the easiest signal to strip or forge.
    synthetic_prob = min(1.0, synthetic_prob + 0.15 * metadata_anomaly)
    manipulation_prob = min(1.0, manipulation_prob + 0.1 * metadata_anomaly)

    model_backed = bool(ai_detection.get("model_backed", False))
    real_model_probability = 1.0 - detector_synthetic_prob
    verified_provenance = bool(provenance.get("verified", False))

    # Missing or neutral evidence is not positive proof that an image is real.
    # Only verified provenance or a confident trained-model real prediction can
    # support the authentic class.
    authenticity_prob = 0.0
    if verified_provenance:
        authenticity_prob = max(authenticity_prob, 0.85)
    if model_backed and real_model_probability >= 0.8:
        authenticity_prob = max(authenticity_prob, real_model_probability)

    scores = {
        "synthetic": synthetic_prob,
        "manipulated": manipulation_prob,
        "authentic": authenticity_prob,
    }
    strong_synthetic = synthetic_prob >= 0.6
    strong_manipulation = manipulation_prob >= 0.6
    strong_authenticity = authenticity_prob >= 0.8
    contradiction = sum((strong_synthetic, strong_manipulation, strong_authenticity)) > 1

    if contradiction:
        verdict = VERDICT_INCONCLUSIVE
        confidence = 0.3
    elif strong_synthetic:
        verdict = VERDICT_AI_GENERATED
        confidence = synthetic_prob
    elif strong_manipulation:
        verdict = VERDICT_MANIPULATED
        confidence = manipulation_prob
    elif strong_authenticity:
        verdict = VERDICT_AUTHENTIC
        confidence = authenticity_prob
    else:
        verdict = VERDICT_INCONCLUSIVE
        confidence = max(synthetic_prob, manipulation_prob, authenticity_prob)

    confidence = round(min(0.97, confidence), 3)
    if confidence >= 0.75:
        confidence_label = "High"
    elif confidence >= 0.5:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "confidence_percent": f"{round(confidence * 100)}%",
        "component_scores": {k: round(v, 3) for k, v in scores.items()},
        "contradiction_detected": contradiction,
        "authenticity_evidence": {
            "trained_model_used": model_backed,
            "verified_provenance": verified_provenance,
        },
    }
