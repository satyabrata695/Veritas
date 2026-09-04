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
    synthetic_prob = ai_detection.get("synthetic_probability", 0.0)
    synthetic_prob = 0.7 * synthetic_prob + 0.3 * noise.get("synthetic_support_score", 0.0)

    manipulation_prob = max(
        face_analysis.get("max_manipulation_probability", 0.0),
        ela.get("manipulation_support_score", 0.0),
        noise.get("splice_support_score", 0.0),
    )

    metadata_anomaly = metadata.get("anomaly_score", 0.0)

    # Provenance can only push toward authenticity, and only when actually
    # verified - an unverified/missing manifest must stay neutral.
    provenance_bonus = 0.0
    if provenance.get("verified"):
        provenance_bonus = 0.3

    # Blend in a light metadata contribution without letting it dominate,
    # since metadata is the easiest signal to strip or forge.
    synthetic_prob = min(1.0, synthetic_prob + 0.15 * metadata_anomaly)
    manipulation_prob = min(1.0, manipulation_prob + 0.1 * metadata_anomaly)

    authenticity_prob = max(0.0, 1.0 - max(synthetic_prob, manipulation_prob) + provenance_bonus)
    authenticity_prob = min(1.0, authenticity_prob)

    scores = {
        "synthetic": synthetic_prob,
        "manipulated": manipulation_prob,
        "authentic": authenticity_prob,
    }
    leading_label, leading_score = max(scores.items(), key=lambda kv: kv[1])
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0

    # Contradiction check: strong signals pointing in different directions
    # should reduce certainty rather than being averaged away silently.
    contradiction = synthetic_prob > 0.5 and manipulation_prob > 0.5 and authenticity_prob > 0.4
    weak_evidence = leading_score < 0.45

    if contradiction or (weak_evidence and margin < 0.1):
        verdict = VERDICT_INCONCLUSIVE
        confidence = round(0.5 - margin / 2, 3)
        confidence_label = "Low"
    else:
        verdict = {
            "synthetic": VERDICT_AI_GENERATED,
            "manipulated": VERDICT_MANIPULATED,
            "authentic": VERDICT_AUTHENTIC,
        }[leading_label]
        confidence = round(min(0.97, leading_score * (0.6 + margin)), 3)
        if confidence >= 0.75:
            confidence_label = "High"
        elif confidence >= 0.5:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"
            if verdict != VERDICT_INCONCLUSIVE and confidence < 0.4:
                verdict = VERDICT_INCONCLUSIVE

    return {
        "verdict": verdict,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "confidence_percent": f"{round(confidence * 100)}%",
        "component_scores": {k: round(v, 3) for k, v in scores.items()},
        "contradiction_detected": contradiction,
    }
