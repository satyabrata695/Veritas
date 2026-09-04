# AI Digital Media Verification System — Hackathon MVP

An evidence-based image-authenticity checker. Instead of a single REAL/FAKE
label, it runs several independent forensic checks in parallel, fuses them
with a calibrated scoring layer, and returns a verdict — **Likely
Authentic**, **Likely AI-Generated**, **Likely Manipulated**, or
**Inconclusive** — with a plain-language explanation and a full evidence
breakdown. This implements the MVP scope from section 13 of the project
proposal.

## What's actually running vs. what's a placeholder

Be upfront about this with judges — it's the project's own key principle
(section 17): detection is probabilistic, and an honest MVP beats an
overconfident one.

| Component | Status |
|---|---|
| Upload → hash → pipeline orchestration | **Real**, working end-to-end |
| Metadata (EXIF) analysis | **Real** |
| Provenance / C2PA check | **Byte-level marker scan only** — not full manifest verification |
| AI-generation detection | **Frequency-domain heuristic** (no trained model) |
| Deepfake / face-manipulation | **Haar-cascade face detection + texture/symmetry heuristics** (no trained model) |
| Error Level Analysis (ELA) | **Real**, JPEG-focused |
| Noise-residual consistency | **Real** |
| Evidence fusion / confidence calibration | **Real**, but weights are hand-set, not learned |
| Explanation layer | **Template-based**, not yet an LLM call |

Every heuristic module (`ai_detector.py`, `face_analysis.py`) is written
behind a small interface specifically so it can be swapped for a trained
model later without touching the API, fusion logic, or frontend. That's
the story to tell judges: *the architecture is production-shaped even
though the MVP detectors are lightweight.*

## Project structure

```
ai-media-verification/
├── backend/
│   ├── main.py                 # FastAPI app — orchestrates the pipeline
│   ├── requirements.txt
│   └── forensics/
│       ├── metadata.py         # Stage 3 — EXIF analysis
│       ├── provenance.py       # Stage 4 — C2PA marker check
│       ├── ai_detector.py      # Stage 5 — AI-generation detection (+ pluggable model interface)
│       ├── face_analysis.py    # Stage 6 — face detection + deepfake heuristics
│       ├── ela.py              # Stage 7a — Error Level Analysis
│       ├── noise.py            # Stage 7b — noise-residual consistency
│       ├── fusion.py           # Stage 8 — evidence fusion / calibration
│       └── report.py           # Stage 9-10 — explanation + final report
└── frontend/
    └── index.html              # Single-file "evidence desk" dashboard (no build step)
```

This mirrors the modular architecture in section 6 of the proposal:
each detector is independent and replaceable, the fusion engine only
talks to structured outputs, and the explanation layer never sees the
raw image or invents findings.

## Running it

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Swagger UI: http://localhost:8000/docs

**Frontend**
Just open `frontend/index.html` in a browser (or serve it with any static
server). It posts to `http://localhost:8000/analyze` by default — editable
in the "Backend endpoint" field in the UI if you deploy the API elsewhere.

**Quick API test**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/image.jpg;type=image/jpeg"
```

This was tested end-to-end in the build environment (server started,
two synthetic test images posted to `/analyze`, verified the response
shape and that a low-noise/gridded test image scored a higher synthetic
probability than a noisy photo-like test image).

## Step-by-step build order (for your hackathon timeline)

1. **Scaffold** — get `main.py` returning a hard-coded JSON report from
   `/analyze` so the frontend has something to render against immediately.
2. **Metadata + provenance** — cheapest wins, no ML dependencies. Wire up
   `metadata.py` and `provenance.py` first.
3. **ELA + noise** — pure PIL/NumPy/OpenCV, no training data needed.
   Gets you two more real signals fast.
4. **AI-generation + face/deepfake detectors** — start with the shipped
   heuristics so the full pipeline runs, then swap in trained models if
   time allows (see "Next steps" below).
5. **Fusion** — once all six modules produce a score, wire `fusion.py`
   and tune the thresholds against a handful of test images by eye.
6. **Frontend polish + demo script** — follow section 15 of the proposal:
   genuine photo → AI portrait → deepfake → recompressed/ambiguous image
   → show it correctly returning "inconclusive".
7. **Evaluation** — even 20-30 labeled images split by source (section 12)
   gives you a real accuracy/precision/recall number to show judges instead
   of just vibes.

## Next steps to strengthen this beyond the MVP

- **Train real detectors** (section 8): fine-tune a ViT/CNN on genuine vs.
  multi-generator synthetic images for `AIGenerationDetector`, and a
  face-forensics model (FaceForensics++-style) for `face_analysis.py`.
  Both modules already expose a `model` swap-in point.
- **Real C2PA verification**: integrate `c2pa-python` (or equivalent) in
  `provenance.py` to validate signed manifests instead of byte-sniffing.
- **Persistence**: replace the in-memory `_ANALYSIS_HISTORY` list in
  `main.py` with PostgreSQL + object storage per section 6, respecting the
  privacy guidance in section 11 (hash-only storage option, retention
  controls, deletion).
- **LLM explanation layer**: send `report.build_report()`'s structured
  output (never the raw image) to an LLM with a strict "summarize, don't
  invent" system prompt, replacing the template in `report.py`.
- **Calibration**: the fusion weights in `fusion.py` are reasonable
  starting points, not measured — recalibrate against the labeled
  evaluation set from section 12 (accuracy, precision/recall, ROC-AUC,
  and actual calibration curves, not just accuracy).

## Key limitation to state to judges

This is a risk-assessment tool, not a fraud-proof detector. AI-media
detection is an arms race — a detector trained today can degrade against
tomorrow's generators. That's why "Inconclusive" is a first-class outcome
here rather than a fallback nobody implements.
