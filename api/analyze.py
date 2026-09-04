"""
Vercel Python Serverless Function — wraps the FastAPI /analyze route.

Vercel invokes this module as an ASGI app. The `app` object at the
bottom of this file is picked up automatically by @vercel/python.

The forensics sub-package lives alongside this file in api/forensics/.
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
import uuid
from pathlib import Path

# Make sure the api/ directory is on sys.path so that
# `from forensics import ...` works inside a Vercel function.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from forensics import ai_detector, ela, face_analysis, fusion, metadata, noise, provenance, report

app = FastAPI(
    title="AI Digital Media Verification System",
    description="Evidence-based authenticity assessment for uploaded images.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

_ANALYSIS_HISTORY: list[dict] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/history")
def history() -> list[dict]:
    return _ANALYSIS_HISTORY[-20:]


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported content type: {file.content_type}")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (15 MB limit).")

    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not decode file as an image.")

    original_format = image.format
    start = time.time()

    meta_result     = metadata.analyze_metadata(image)
    prov_result     = provenance.analyze_provenance(raw_bytes)
    ai_result       = ai_detector.AI_DETECTOR.run(image)
    face_result     = face_analysis.analyze_faces(image)
    ela_result      = ela.analyze_ela(image, original_format)
    noise_result    = noise.analyze_noise(image)

    fused = fusion.fuse_evidence(
        metadata=meta_result,
        provenance=prov_result,
        ai_detection=ai_result,
        face_analysis=face_result,
        ela=ela_result,
        noise=noise_result,
    )

    final_report = report.build_report(
        file_hash=file_hash,
        original_format=original_format,
        metadata=meta_result,
        provenance=prov_result,
        ai_detection=ai_result,
        face_analysis=face_result,
        ela=ela_result,
        noise=noise_result,
        fused=fused,
    )

    final_report["analysis_id"] = str(uuid.uuid4())
    final_report["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    final_report["filename"] = file.filename

    _ANALYSIS_HISTORY.append({
        "analysis_id": final_report["analysis_id"],
        "filename": file.filename,
        "verdict": final_report["verdict"],
        "confidence_percent": final_report["confidence_percent"],
        "file_hash_sha256": file_hash,
    })

    return final_report
