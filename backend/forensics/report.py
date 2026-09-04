"""
Stage 9-10: Explanation + final report.

Per the proposal, the explanation layer receives structured evidence and
is NOT allowed to invent findings or override the detector/fusion output.
This template-based version guarantees that constraint by construction.

To upgrade this to an LLM-written explanation (as section 9 of the
proposal suggests): send `build_report()`'s output as read-only structured
context to a model with a system prompt like "Summarize this evidence in
plain language. Do not add, remove, or reinterpret any finding, number, or
verdict below." Never let the model see the raw image or invent new scores.
"""
from __future__ import annotations

from typing import Any


def _plain_language_summary(fused: dict[str, Any], evidence_flags: list[str]) -> str:
    verdict = fused["verdict"]
    conf = fused["confidence_percent"]

    openers = {
        "LIKELY AI-GENERATED": f"This image is likely AI-generated ({conf} confidence).",
        "LIKELY MANIPULATED": f"This image appears to have been manipulated or edited ({conf} confidence).",
        "LIKELY AUTHENTIC": f"This image is likely an authentic, unmanipulated photo ({conf} confidence).",
        "INCONCLUSIVE": "The evidence is mixed or too weak to reach a confident conclusion.",
    }
    summary = openers.get(verdict, verdict)

    if fused.get("contradiction_detected"):
        summary += " Signals from different detectors pointed in conflicting directions, which lowered overall confidence."

    if evidence_flags:
        top = evidence_flags[:3]
        summary += " Key evidence: " + "; ".join(top) + "."
    else:
        summary += " No strong individual red flags were found by any single detector."

    return summary


def build_report(
    file_hash: str,
    original_format: str | None,
    metadata: dict[str, Any],
    provenance: dict[str, Any],
    ai_detection: dict[str, Any],
    face_analysis: dict[str, Any],
    ela: dict[str, Any],
    noise: dict[str, Any],
    fused: dict[str, Any],
) -> dict[str, Any]:
    all_flags = (
        metadata.get("flags", [])
        + ela.get("flags", [])
        + noise.get("flags", [])
        + ai_detection.get("flags", [])
        + face_analysis.get("flags", [])
    )

    evidence_breakdown = [
        {
            "category": "AI-generation forensics",
            "score": ai_detection.get("synthetic_probability", 0.0),
            "detail": f"Method: {ai_detection.get('method')}",
            "flags": ai_detection.get("flags", []),
        },
        {
            "category": "Face / deepfake analysis",
            "score": face_analysis.get("max_manipulation_probability", 0.0),
            "detail": f"{face_analysis.get('faces_detected', 0)} face(s) detected",
            "flags": face_analysis.get("flags", []),
        },
        {
            "category": "Compression / Error Level Analysis",
            "score": ela.get("manipulation_support_score", 0.0),
            "detail": "Applicable" if ela.get("applicable") else "Low confidence (non-JPEG source)",
            "flags": ela.get("flags", []),
        },
        {
            "category": "Noise-residual consistency",
            "score": max(noise.get("synthetic_support_score", 0.0), noise.get("splice_support_score", 0.0)),
            "detail": f"Variability: {noise.get('noise_variability')}",
            "flags": noise.get("flags", []),
        },
        {
            "category": "Metadata",
            "score": metadata.get("anomaly_score", 0.0),
            "detail": metadata.get("software_tag") or "No software tag",
            "flags": metadata.get("flags", []),
        },
        {
            "category": "Provenance (C2PA)",
            "score": 1.0 if provenance.get("verified") else 0.0,
            "detail": provenance.get("note"),
            "flags": [],
        },
    ]

    return {
        "file_hash_sha256": file_hash,
        "source_format": original_format,
        "verdict": fused["verdict"],
        "confidence": fused["confidence"],
        "confidence_label": fused["confidence_label"],
        "confidence_percent": fused["confidence_percent"],
        "component_scores": fused["component_scores"],
        "evidence_breakdown": evidence_breakdown,
        "summary": _plain_language_summary(fused, all_flags),
        "limitation": (
            "This is an AI-assisted risk assessment, not proof of authenticity or fraud. "
            "Detectors trained today can perform worse against future or unseen generators."
        ),
        "raw": {
            "metadata": metadata,
            "provenance": provenance,
            "ai_detection": ai_detection,
            "face_analysis": face_analysis,
            "ela": ela,
            "noise": noise,
        },
    }
