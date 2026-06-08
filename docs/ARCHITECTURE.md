# Architecture

## Goal & constraints

Show creators the *predicted* neural response to their content, cheaply and honestly.

- **Model:** TRIBE v2 (V-JEPA2 video + Wav2Vec-BERT audio + Llama 3.2-3B text → ~1B-param transformer)
  predicts an average human's fMRI response: `(n_timesteps, ~20k vertices)` on the **fsaverage5** mesh,
  offset 5 s for hemodynamic lag. Used **as-is**.
- **License:** CC-BY-NC-4.0 → non-commercial only; Llama 3.2 encoder is gated (needs an HF token).
- **Budget:** ~$100. A GPU can't run 24/7, so inference is **serverless and scales to zero**.
- **Hardware reality:** TRIBE needs a real GPU (≥16-24 GB VRAM). The dev laptop (RTX 2050, 4 GB) is
  for coding only.

## Data flow

1. Browser uploads a file to `POST /api/analyze` on the Flask backend.
2. Backend validates type/size, stores the file (R2 or local), and **spawns** the Modal GPU worker
   (`inference_client.spawn` → `TribeWorker.analyze_bytes.spawn`). It returns a `job_id` immediately
   (HTTP 202) with a Modal call id recorded in the job file.
3. Frontend polls `GET /api/jobs/:id`. The backend non-blockingly polls the Modal call
   (`FunctionCall.get(timeout=0)`); while pending it returns `{status: "running"}`.
4. The worker: routes modality → `get_events_dataframe(**kwargs)` → `model.predict()` →
   `scoring.score_predictions()` → `insights.generate_insights()` → result dict.
5. When ready, the backend caches the result (R2/local) and serves it. Frontend renders charts +
   insight cards.

## Why these boundaries

- **Backend ≠ GPU.** Keeping orchestration on a $6 CPU box means the expensive GPU only runs during the
  seconds of actual inference. The backend never blocks on it.
- **JSON contract (`shared/result_schema.json`)** is the only coupling between tiers, so each can be
  developed/deployed independently. `MOCK_INFERENCE=1` lets the frontend/backend run with zero GPU cost.
- **Modal Volume for weights.** Multi-GB backbones download once into a persistent Volume, so warm and
  subsequent cold starts skip the download.

## Scoring pipeline (inference/)

```
predict() → (T, V) raw activation on fsaverage5
   │  networks.py: boolean vertex masks per functional network (Yeo-7 + Destrieux atlases)
   ▼
per-network |activation| aggregated over vertices × time
   │  scoring.py: z-score vs reference distribution → logistic → 0-100
   ▼
network scores ──weighted blend (NETWORK_META weights)──> composite Neural Engagement Index
network scores per timestep ──resample to 1 Hz──> engagement timeline
   │  insights.py: rule-based cards (hook/pacing/captions/emotion)
   ▼
result dict (shared schema)
```

The six networks (visual, auditory, language, emotional_social, default_mode, multisensory) are exactly
the ones TRIBE v2's paper validates in Figs. 4-7. Composite-index weights are explicit in
`NETWORK_META` so the score is explainable, not a black box.

## Calibration

`scoring.py` normalizes each network against a reference mean/std so "typical" content lands near 50.
Ship-blocking nuance: the defaults are placeholders. Run `scoring.build_reference([...preds])` over a
handful of representative creator clips (offline, via the Modal worker) to write real `REFERENCE_STATS`
before exposing scores to users.

## Known limitations

- `image` and `text` inputs are degenerate for a temporal video+audio+text model → flagged
  `experimental` and labeled as such in the UI.
- fMRI has ~second-scale temporal resolution; the model captures *where*, not millisecond dynamics.
- "Engagement" here is a neuro-proxy, not a validated predictor of social-media reach.
