# VERITAS

### Personal project · AI-generated and manipulated image analysis

An evidence-based image-authenticity checker I built to explore practical
approaches to detecting AI-generated and manipulated media. Rather than
returning a single `REAL` or `FAKE` label, it runs several independent
forensic checks, combines their evidence with a calibrated scoring layer, and
returns one of four outcomes: **Likely Authentic**, **Likely AI-Generated**,
**Likely Manipulated**, or **Inconclusive**. Each result includes a
plain-language explanation and an evidence breakdown.

## Project overview

### Current implementation

This is a personal MVP and an ongoing learning project. Some components are
fully implemented, while others deliberately use lightweight heuristics that
can later be replaced with trained models or standards-compliant services.

| Component | Current implementation |
|---|---|
| Upload, hashing, and pipeline orchestration | Working end-to-end |
| Metadata (EXIF) analysis | Implemented |
| Provenance / C2PA check | Byte-level marker scan; not full manifest verification |
| AI-generation detection | Frequency-domain heuristic; no trained model |
| Face-manipulation analysis | Haar-cascade face detection with texture and symmetry heuristics |
| Error Level Analysis (ELA) | Implemented for JPEG images |
| Noise-residual consistency | Implemented |
| Evidence fusion and confidence calibration | Implemented with hand-tuned weights |
| Explanation layer | Template-based; not an LLM integration |

The heuristic modules (`ai_detector.py` and `face_analysis.py`) use small,
replaceable interfaces so trained models can be integrated later without
changing the API, fusion logic, or frontend.

## Technical design

### Project structure

```text
VERITAS/
├── backend/
│   ├── main.py                 # FastAPI app and pipeline orchestration
│   ├── requirements.txt
│   └── forensics/
│       ├── metadata.py         # EXIF analysis
│       ├── provenance.py       # C2PA marker check
│       ├── ai_detector.py      # AI-generation detection
│       ├── face_analysis.py    # Face detection and deepfake heuristics
│       ├── ela.py              # Error Level Analysis
│       ├── noise.py            # Noise-residual consistency
│       ├── fusion.py           # Evidence fusion and calibration
│       └── report.py           # Explanation and final report
└── frontend/
    └── index.html              # Single-file dashboard
```

Each detector is independent and replaceable. The fusion engine consumes
structured detector outputs, and the explanation layer reports only the
evidence it receives.

## Get started

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API documentation is available at <http://localhost:8000/docs>.

### Frontend

Open `frontend/index.html` in a browser, or serve it with any static server.
By default, it sends requests to `http://localhost:8000/analyze`. The endpoint
can be changed in the dashboard if the API is deployed elsewhere.

### Quick API test

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/image.jpg;type=image/jpeg"
```

## Roadmap

- Replace the AI-generation and face-manipulation heuristics with trained
  models.
- Integrate standards-compliant C2PA manifest and signature verification.
- Add persistent storage with privacy-focused retention and deletion controls.
- Generate explanations from structured evidence with an LLM constrained to
  summarize rather than invent findings.
- Calibrate fusion weights against a labeled evaluation set and report
  precision, recall, ROC-AUC, and calibration metrics.

## Important limitations

This project provides a risk assessment, not proof of authenticity or fraud.
AI-media detection changes rapidly as generation methods evolve, so
**Inconclusive** is intentionally a first-class outcome. Results should be
used as one input to a broader verification process, especially for
high-stakes decisions.
