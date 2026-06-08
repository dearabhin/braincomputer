# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**BrainComputer.in** — a non-commercial web tool that runs creator content (video / image /
audio / text) through Meta's **TRIBE v2** fMRI brain-encoding model and returns neural-engagement
proxy scores, charts, and plain-language insights. The model is used **as-is** (no retraining).

> **License constraint that shapes everything:** TRIBE v2 is **CC-BY-NC-4.0 (non-commercial)** and
> its Llama 3.2 encoder is gated. This project is intentionally **free / portfolio / research** — do
> **not** add paid tiers, ads, or any commercial use without written permission from Meta. Keep the
> "predicted brain activity, not a guarantee of reach" disclaimer visible in the UI.

## Architecture (3 decoupled tiers)

```
Browser ──upload──> Cloudflare Pages (static frontend/)
                         │  POST /api/analyze, poll GET /api/jobs/:id
                         ▼
                   Flask API (backend/)  — CPU only, pure orchestration
                         │  stores upload (R2/local), spawns Modal worker, polls, caches result
                         ▼
                   Modal GPU worker (inference/)  — loads TRIBE v2, predicts, scores
                         ▼
                   result JSON (shared/result_schema.json) ──> backend ──> browser
```

The tiers communicate only through the JSON contract in **`shared/result_schema.json`** — when you
change the result shape, update that file, the inference worker, and the frontend renderer together.

### `inference/` — the GPU worker (the heart of the system)
- `modal_app.py` — Modal app. `TribeWorker` loads `TribeModel.from_pretrained("facebook/tribev2")`
  once per container (`@modal.enter`), with weights cached in a Modal **Volume** at `/cache` and the
  HF token supplied via a Modal **secret** named `huggingface`. `analyze_bytes(data, filename, job_id)`
  is the entrypoint: bytes → result dict.
- `preprocess.py` — modality routing. video→`video_path`, audio→`audio_path`, text→`text_path`
  (TRIBE does TTS+timing). **Image is not native**: it's rendered to a ~2s silent mp4 via ffmpeg and
  fed as `video_path` (mirrors the paper's flashed-image localizer). image & text are flagged
  `experimental`.
- `networks.py` — builds boolean vertex masks on the **fsaverage5** mesh (~20,484 verts) for six
  functional networks (visual, auditory, language, emotional_social, default_mode, multisensory)
  from nilearn atlases (Yeo-7 + Destrieux). Cached to `NETWORKS_CACHE`. **Network weights for the
  composite index live in `NETWORK_META` here** and are intentionally transparent.
- `scoring.py` — `score_predictions(preds, duration_s)` turns the raw `(n_timesteps, n_vertices)`
  prediction into per-network 0-100 scores (logistic on a z-score vs a reference distribution),
  the composite Neural Engagement Index, and a 1 Hz timeline. `build_reference()` calibrates the
  reference distribution from sample clips → `REFERENCE_STATS`.
- `insights.py` — deterministic, rule-based cards from the score profile + timeline. No second model.

### `backend/` — Flask API (no GPU)
- `app.py` — `POST /api/analyze` (multipart `file`), `GET /api/jobs/<id>`, `GET /healthz`. Jobs are
  tiny JSON records in `JOBS_DIR`; results cached via `storage`.
- `inference_client.py` — `spawn()` / `poll()` against the Modal worker. **`MOCK_INFERENCE=1`** skips
  Modal entirely and returns a deterministic fake result — use this for frontend/backend dev with no GPU.
- `storage.py` — R2 (S3-compatible) when `R2_*` env is set, else local disk. Same interface.
- `config.py` — all settings are env-driven (see `.env.example`).

### `frontend/` — Cloudflare Pages (static, vanilla JS, no build step)
- `index.html` (upload, drag-drop) → `results.html` (dashboard). `js/config.js` resolves the API base
  (`<meta name="bc-api-base">` or localhost in dev). `js/charts.js` uses Chart.js via CDN.

## Commands

```bash
# Scoring/insights unit smoke test (no GPU, no model, no cloud):
python3 tools/smoke_scoring.py

# Backend locally in mock mode (no GPU):
cd backend && MOCK_INFERENCE=1 python app.py        # serves :8000
# Frontend locally:
python3 -m http.server 8099 --directory frontend    # or: wrangler pages dev frontend

# Real GPU inference (requires Modal account + `huggingface` secret + HF access to Llama 3.2):
modal run inference/modal_app.py --input sample_reel.mp4   # smoke test one file
modal deploy inference/modal_app.py                        # deploy the worker

# Deploy frontend:
wrangler pages deploy frontend --project-name=braincomputer
```

There is no automated test suite yet; `tools/smoke_scoring.py` is the fastest correctness check for the
scoring path and validates output against `shared/result_schema.json` when `jsonschema` is installed.

## Conventions / gotchas

- **Keep the three tiers runnable in isolation.** `inference/` modules import each other by bare name
  (`import networks`) because Modal adds them as local sources at the package root — don't convert them
  to package-relative imports.
- **Never block the HTTP request on GPU work.** The backend spawns and polls; the frontend polls
  `GET /api/jobs/:id`. Cold Modal starts can take a minute (weight download on first run only).
- **Honesty is a product requirement, not decoration.** Scores are proxies from *predicted* brain
  activity. Don't add copy that promises views/reach. `image`/`text` inputs stay labeled experimental.
- The reference distribution in `scoring.py` ships with conservative placeholders; run `build_reference()`
  over a handful of representative clips to make the 0-100 scale meaningful before showing real users.
- `meta_tribe_v2.pdf` (the paper) lives in `docs/` — the functional networks in `networks.py` map to the
  ones it validates (Figs. 4-7).
