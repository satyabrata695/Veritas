# Build Guide: AI Digital Media Verification System

This is the full path from "MVP with heuristics" to a real, defensible
project. Each phase builds on the last — don't skip to model training
before the pipeline and data are solid, or you'll be debugging two
things at once.

Rough total timeline if working part-time: **6-10 weeks**. Phases 1-2 can
be done in a weekend; phases 3-4 (real models) are the long pole.

---

## Phase 0 — Environment & accounts (Day 1)

1. **Install tooling**: Python 3.11+, Node.js 18+ (only if you build a
   real React frontend later), Docker, Git.
2. **Get a GPU story sorted early** — you cannot train image classifiers
   on CPU in reasonable time. Pick one:
   - Free/cheap: Google Colab (free T4, or Colab Pro for A100 access),
     Kaggle Notebooks (free 30 GPU-hrs/week).
   - Paid on-demand: RunPod, Lambda Labs, Vast.ai (~$0.20-0.50/hr for a
     3090/4090).
   - If you have a local NVIDIA GPU with 8GB+ VRAM, that works too.
3. **Create accounts you'll likely need**: Hugging Face (datasets +
   model hosting), Weights & Biases (experiment tracking, free tier),
   a cloud provider (Railway/Render/Fly.io for the API, or AWS/GCP if
   you want more control).
4. **Set up version control**: push the existing codebase to a private
   GitHub repo now, before you touch anything. Commit after every phase.

---

## Phase 1 — Get the existing MVP running end-to-end (Day 1-2)

You already have working code for this. Do it now so you have a
baseline to compare against as you improve things.

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` in a browser, upload a few images (a real
phone photo, a screenshot, an AI-generated image from any generator you
have access to), and read the JSON each detector produces. Understand
what each module is currently doing before improving it — read
`backend/forensics/*.py`, every file has a docstring explaining its
method and honest limitations.

**Checkpoint:** you should be able to explain, in your own words, why
each of the six detectors exists and what specific manipulation pattern
it's meant to catch.

---

## Phase 2 — Build a real evaluation dataset (Week 1-2)

Do this **before** training any model. Without labeled data you can't
tell if a "detector" works, and every later phase depends on it.

### 2.1 — Collect authentic images
- Public domain / permissively licensed photo datasets: **RAISE**
  (uncompressed camera originals, built for forensics), **Flickr30k**,
  or your own phone photos (best — you know the ground truth and get
  real EXIF).
- Aim for variety: multiple cameras/phones, indoor/outdoor, with and
  without people.

### 2.2 — Collect AI-generated images
- Generate your own across multiple tools so your detector doesn't
  overfit to one generator's fingerprint (the proposal calls this out
  explicitly in section 8A): Midjourney, DALL-E 3, Stable Diffusion
  (SDXL/SD3), Flux, Firefly.
- Or use existing labeled datasets: **GenImage**, **DiffusionDB**,
  **ArtiFact**.

### 2.3 — Collect manipulated/deepfake images
- **FaceForensics++** — the standard academic dataset for face-swap
  and reenactment deepfakes, widely used to train and benchmark
  detectors.
- **Celeb-DF** — higher-quality deepfakes, harder detection benchmark.
- Make some of your own simple manipulations too (splice, clone-stamp,
  face-swap with an open tool) so you understand failure modes
  first-hand.

### 2.4 — Structure it for evaluation, not just training
Per section 12 of the proposal, **structure by task and split**:
```
dataset/
├── ai_image_detector/
│   ├── train/
│   │   ├── authentic/{raise, mobile}
│   │   └── ai_generated/{flux, midjourney, sdxl}
│   ├── validation/
│   │   ├── authentic/
│   │   └── ai_generated/
│   └── test/
│       ├── authentic/
│       └── ai_generated/
├── deepfake_detector/
│   ├── train/{real, fake}
│   ├── validation/{real, fake}
│   └── test/{real, fake}
└── manifests/
    └── RAISE_16.csv
```
The unseen-generator test set is what actually proves robustness —
without it you'll get a great accuracy number that means nothing.

### 2.5 — Write a labels manifest
A CSV/JSON mapping `filepath → label → source/generator → split`. Every
later phase (training, evaluation, calibration) reads from this one
file — build it once, carefully.

**Checkpoint:** 2,000+ images minimum across all classes to start
(more is better), with an unseen-generator held-out set of at least a
few hundred images.

---

## Phase 3 — Replace the AI-generation heuristic with a trained model (Week 2-4)

This replaces `backend/forensics/ai_detector.py`'s heuristic with a
real classifier, using the `AIGenerationDetector` interface already
built for this swap.

1. **Pick an architecture.** Don't start from scratch:
   - Fine-tune a pretrained **ViT** or **ConvNeXt** (via `timm` or
     Hugging Face `transformers`) as a binary classifier
     (authentic vs. synthetic). This is the standard, well-documented
     approach and a good default.
   - For stronger generalization to unseen generators, look at
     **CLIP-based detectors** (e.g. fine-tuning a linear probe on CLIP
     features) — research (Ojha et al., "Universal Fake Image
     Detectors") shows these generalize better across generators than
     training a CNN end-to-end.
2. **Train**, tracking metrics in W&B: start with a small subset to
   verify the training loop works, then scale to the full dataset.
   Use data augmentation that mimics real-world degradation (JPEG
   recompression, resizing, slight blur) — models trained only on
   clean images collapse the moment an image has been through
   WhatsApp/Instagram once.
3. **Evaluate on the held-out unseen-generator set specifically.** This
   number is the one worth presenting; the same-generator accuracy will
   look inflated.
4. **Export** the trained weights (ONNX or TorchScript for a lighter
   serving footprint) and load them in a new class implementing the
   `TrainedClassifier` protocol already defined in `ai_detector.py`:
   ```python
   class MyTrainedModel:
       def predict_proba(self, image: Image.Image) -> float:
           # preprocess -> run inference -> return synthetic probability
           ...
   ```
5. Swap it in: `AIGenerationDetector(model=MyTrainedModel())` in
   `main.py`. Nothing else in the pipeline changes.

**Checkpoint:** report accuracy, precision, recall, F1, and ROC-AUC
separately for seen and unseen generators. Expect a real gap between
them — that gap *is* your headline "arms race" finding (proposal
section 17), not a failure.

---

## Phase 4 — Replace the deepfake heuristic with a trained model (Week 3-5, can overlap Phase 3)

Same pattern, applied to `backend/forensics/face_analysis.py`.

1. Train (or fine-tune a published checkpoint of) a face-forensics
   model on FaceForensics++/Celeb-DF — **Xception** fine-tuned on face
   crops is the classic, well-benchmarked baseline; **EfficientNet** or
   a lightweight ViT are solid alternatives.
2. Keep the existing face-detection + crop/align step (Haar cascade is
   fine for an MVP, but consider swapping to **MTCNN** or
   **RetinaFace** for more robust detection on varied poses/lighting).
3. Wrap the trained model behind the same kind of swap-in interface, so
   `analyze_faces()` calls the model instead of
   `_heuristic_manipulation_score()` when a model is provided.
4. Evaluate per manipulation type (face-swap vs. reenactment vs. GAN
   synthesis) — deepfake detectors often do well on one type and badly
   on another, and that breakdown is worth showing.

---

## Phase 5 — Recalibrate the fusion engine on real data (Week 5)

The weights in `fusion.py` were hand-picked for the MVP. Now that you
have real detector outputs and ground-truth labels:

1. Run every detector over your full labeled dataset and log all raw
   scores.
2. Fit the fusion weights properly instead of guessing — a simple
   **logistic regression** over `[synthetic_prob, manipulation_prob,
   metadata_anomaly, provenance_verified]` as features, trained to
   predict the true label, is a huge, easy upgrade over hand-set
   weights and keeps the "explainable evidence" property intact (you
   can still show each input score to the user).
3. **Calibrate the confidence**, don't just classify: use
   `sklearn.calibration.calibration_curve` and, if needed, Platt
   scaling or isotonic regression so that "80% confidence" actually
   corresponds to ~80% correctness empirically. This is called out
   explicitly in proposal section 12 and is often skipped — doing it
   properly is a real differentiator.
4. Re-tune the "Inconclusive" thresholds (currently in `fusion.py`)
   against your validation set: what margin/score actually predicts
   "the model is likely to be wrong here"?

---

## Phase 6 — Real provenance (C2PA) verification (Week 5-6, can be done anytime)

Replace the byte-sniffing placeholder in `provenance.py`:

1. Look at `c2pa-python` (official bindings) or the C2PA reference
   tooling to actually parse and validate embedded manifests —
   signature verification, trust-chain checking against known
   certificate authorities, and reading the edit-history assertions.
2. Generate your own test images with Content Credentials (Adobe's
   tools and some camera apps support this now) so you have real
   positive test cases, since most images in the wild still won't have
   a manifest.
3. Keep the honest framing: verified provenance is strong positive
   evidence; absence is neutral, never negative.

---

## Phase 7 — Persistence & backend hardening (Week 6)

1. **Database**: replace the in-memory `_ANALYSIS_HISTORY` list in
   `main.py` with PostgreSQL. Minimal schema: `analyses(id, file_hash,
   filename, verdict, confidence, created_at, raw_report_json)`. Use
   SQLAlchemy or SQLModel.
2. **Object storage** (S3/GCS/R2) if you decide to retain uploaded
   images — otherwise store only the hash, per the privacy-first option
   in proposal section 11.
3. **Async job queue** for slower model inference: Redis + Celery/RQ
   so `/analyze` can return a job ID immediately and the frontend polls
   or gets a websocket push, instead of holding the HTTP connection
   open for a multi-second inference.
4. **Rate limiting & auth**: at minimum an API key per client; add
   `slowapi` or similar for basic rate limiting before this is public.
5. **Lock down CORS** (`main.py` currently allows `*` — fine for local
   dev, not for anything deployed).
6. **Input validation hardening**: file-type sniffing beyond
   `Content-Type` header (e.g. `python-magic`), size limits already in
   place, and image-bomb protection (`PIL.Image.MAX_IMAGE_PIXELS`).

---

## Phase 8 — Explanation layer upgrade (Week 6-7, optional but high-impact)

Replace the template text in `report.py` with an actual LLM call:

1. Build a strict system prompt: *"You will be given structured
   forensic evidence as JSON. Summarize it in plain language for a
   non-technical user. Do not add, remove, invent, or reinterpret any
   score, finding, or verdict. If evidence is weak, say so."*
2. Pass only the structured JSON `report.build_report()` already
   produces — never the raw image, and never let the LLM see anything
   it could hallucinate additional "evidence" from.
3. Cache explanations by a hash of the evidence JSON so repeated/similar
   reports don't re-call the LLM.
4. Keep the template-based version as a fallback if the LLM call fails
   or times out — the system should never block on this.

---

## Phase 9 — Frontend upgrade (Week 7, optional)

The current `frontend/index.html` is a fine working demo. If you want a
production-shaped frontend:

1. Rebuild in React/Next.js per the original architecture doc, with the
   same visual design language (case-file/evidence-desk aesthetic) —
   reuse the design tokens already established (colors, IBM Plex Mono +
   Newsreader pairing).
2. Add a **history page** backed by the new database instead of
   in-memory/session-only history.
3. Add **image region highlighting**: overlay the face bounding boxes
   and ELA hotspot blocks the backend already computes, directly on the
   uploaded image, so "evidence" is visually pointed to, not just
   listed as text.
4. Add drag-and-drop batch upload for testing multiple images at once.

---

## Phase 10 — Evaluation write-up & docs (ongoing, finalize Week 8)

This is what turns "a working demo" into "a project people trust":

1. Full metrics table: accuracy, precision, recall, F1, ROC-AUC,
   PR-AUC, and calibration error — each split by seen vs. unseen
   generators and by manipulation type.
2. A **limitations section** that's specific, not just "it might be
   wrong sometimes" — e.g. "recall on unseen diffusion generators drops
   from 91% to 62%," "performance degrades after 2+ rounds of
   re-compression," "the deepfake detector was not evaluated on
   non-frontal faces."
3. A short **model card** for each trained model (training data,
   intended use, known failure modes) — standard practice, and easy to
   point to if anyone asks how it was built.
4. Update the README's "what's real vs. placeholder" table — by the end
   of this guide, almost everything in that table should say "Real."

---

## Phase 11 — Deployment (Week 8-9)

1. **Containerize**: a `Dockerfile` for the backend (base image with
   CUDA if you're serving GPU inference, CPU-only is fine and cheaper
   if your models are small/quantized).
2. **Deploy**: Railway/Render/Fly.io for a simple always-on API, or a
   serverless GPU endpoint (Modal, Replicate, Banana) if traffic is
   bursty and you don't want to pay for idle GPU time.
3. **Serve the frontend** as a static site (Vercel/Netlify/Cloudflare
   Pages) pointing at the deployed API URL.
4. **Monitoring**: basic uptime + error logging (Sentry is a fast add),
   and log verdict distribution over time so you notice if the model
   starts drifting.

---

## Suggested order if you only have limited time

If you have to cut scope, prioritize in this order — each item is a
bigger credibility jump than the next:

1. Real evaluation dataset (Phase 2) — without this, nothing else is
   measurable.
2. Trained AI-generation classifier (Phase 3) — the single most
   convincing upgrade over the current heuristic.
3. Calibrated fusion (Phase 5) — cheap, and directly addresses the
   proposal's own stated design principle.
4. Everything else, roughly in the order above.

---

## Where you are right now

You have Phase 0-1 done (working pipeline, all six detector slots
wired up, fusion engine, dashboard). The natural next concrete action
is **Phase 2: start collecting/organizing your dataset** — that's the
one piece nothing else can substitute for.
