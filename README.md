# 🧠 BrainComputer.in

A non-commercial web tool that shows content creators how the human brain is *predicted* to respond
to their short-form content. Upload a **video, image, audio clip, or caption**; we run it through
Meta's [**TRIBE v2**](https://github.com/facebookresearch/tribev2) fMRI brain-encoding model and
return neural-engagement proxy scores, brain-network charts, an engagement timeline, and plain-language
tips on hooks, pacing, captions, and emotional pull.

> ⚠️ **Honest framing.** TRIBE v2 predicts *where an average human brain activates* — it does **not**
> measure real viewers or guarantee reach/virality. Scores here are neuro-derived **proxies** meant to
> guide creative choices.
>
> ⚖️ **License.** TRIBE v2 is **CC-BY-NC-4.0 (non-commercial)** and its Llama 3.2 encoder is gated.
> This project is **free / research / portfolio** by design. Don't add paid tiers or other commercial
> use without written permission from Meta.

## How it works

```
Browser → Cloudflare Pages (static) → Flask API (CPU) → Modal serverless GPU (TRIBE v2) → scores → back
```

- **Frontend** (`frontend/`): static HTML/CSS/vanilla JS on Cloudflare Pages (free). Chart.js via CDN.
- **Backend** (`backend/`): Flask orchestration on a small CPU droplet — stores the upload, hands it to
  the GPU worker, polls, and serves the result. No GPU.
- **Inference** (`inference/`): a Modal serverless-GPU worker that loads TRIBE v2, predicts brain
  activity on the fsaverage5 mesh, and turns it into scores/insights. Scales to zero between uploads,
  so a ~$100 budget lasts.

See [`CLAUDE.md`](CLAUDE.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## Quick start (no GPU needed)

```bash
# 1. Scoring/insights smoke test
python3 tools/smoke_scoring.py

# 2. Backend in mock mode
cd backend && pip install -r requirements.txt
MOCK_INFERENCE=1 python app.py            # http://localhost:8000

# 3. Frontend (new terminal)
python3 -m http.server 8099 --directory frontend
# open http://localhost:8099
```

`MOCK_INFERENCE=1` returns deterministic fake results so you can build the whole UX without a GPU.

## Running real inference

Requires a [Modal](https://modal.com) account and Hugging Face access to the gated Llama 3.2 model.

```bash
pip install modal && modal token new
modal secret create huggingface HF_TOKEN=hf_xxx     # token with Llama 3.2 access
modal run inference/modal_app.py --input sample_reel.mp4   # smoke test
modal deploy inference/modal_app.py                        # deploy the worker
```

Then run the backend **without** `MOCK_INFERENCE` (with `modal token` configured in its environment).

## Deploy

**New to this? Follow the complete beginner walkthrough: [`DEPLOYMENT.md`](DEPLOYMENT.md).** It covers
GitHub, Hugging Face, Modal, the DigitalOcean droplet, every VPS command, and all Cloudflare settings.

Quick reference:

| Tier      | Where                | Notes |
|-----------|----------------------|-------|
| Frontend  | Cloudflare Pages     | `wrangler pages deploy frontend --project-name=braincomputer` |
| Backend   | DO droplet / Azure B1s | `docker build -t bc-api backend && docker run` (see `backend/Dockerfile`) |
| Inference | Modal                | `modal deploy inference/modal_app.py` (scales to zero) |
| Storage   | Cloudflare R2        | set `R2_*` env on the backend |
| DNS       | Cloudflare           | apex/`www` → Pages, `api.` → backend droplet |

## Credits

Built on **TRIBE v2** — d'Ascoli et al., *A foundation model of vision, audition, and language for
in-silico neuroscience* (2026). Used as-is under CC-BY-NC-4.0. Not affiliated with Meta.
