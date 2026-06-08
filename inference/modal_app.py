"""Modal serverless-GPU worker that runs TRIBE v2 and returns engagement scores.

Why Modal: TRIBE v2 stacks V-JEPA2 (video), Wav2Vec-BERT (audio) and Llama 3.2-3B
(text) into a ~1B-parameter transformer, which needs a real GPU. Modal lets the
function scale to zero between uploads so a tight (~$100) budget lasts -- you only
pay for the seconds a clip is actually being analyzed.

Design:
  - Weights download once into a Modal Volume (``/cache``) and persist across cold
    starts, so subsequent runs skip the multi-GB download.
  - The gated Llama 3.2 encoder needs a Hugging Face token, provided as a Modal
    secret named ``huggingface``.
  - ``analyze_file`` is the entrypoint: bytes in -> result dict (shared schema) out.

Deploy:   modal deploy inference/modal_app.py
Local run: modal run inference/modal_app.py --input sample_reel.mp4

NOTE: TRIBE v2 is CC-BY-NC-4.0. This worker is for the non-commercial BrainComputer
project; do not deploy it behind a paid product without Meta's written permission.
"""

from __future__ import annotations

import os
import time

import modal

CACHE_DIR = "/cache"
MODEL_ID = "facebook/tribev2"

# --- Image: system ffmpeg + python deps incl. the tribev2 package from GitHub. ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        # Pin exca to the version neuralset 0.0.2 (a TRIBE dep) was built against.
        # neuralset calls exca.steps.base.NoValue(), but newer exca (>=~0.5.23) moved
        # NoValue to exca.steps.identity, so the unpinned newest release (0.5.26)
        # breaks `import TribeModel` with AttributeError. 0.5.20 satisfies neuralset's
        # `exca>=0.5.20` floor and still has NoValue in exca.steps.base.
        "exca==0.5.20",
        # Pin transformers to the version current at TRIBE's release (neuralset 0.0.2,
        # 2026-03-25). TRIBE leaves transformers UNPINNED, so pip grabbed the newest
        # (5.10.2), where the V-JEPA2 video processor's `AutoProcessor` import path
        # changed -> "Could not import module 'AutoProcessor'". 5.3.0 (2026-03-04) was
        # the current release on TRIBE's release day (5.4.0 came 2026-03-27).
        "transformers==5.3.0",
        # TRIBE v2 itself. Its pyproject pins compatible torch/numpy/pandas, but leaves
        # transformers + exca loose (hence the two pins above). We pin numpy/torch via
        # TRIBE's own constraints, so we don't list them here.
        "git+https://github.com/facebookresearch/tribev2.git",
        # nilearn is only in TRIBE's optional [plotting] extra, but inference/networks.py
        # needs it for the functional-network atlases — install it explicitly.
        "nilearn",
        # nltk: TRIBE transcribes video/audio speech with whisperx, whose word-alignment
        # needs NLTK's `punkt_tab` data. We install nltk here only to pre-download that
        # data into the image (next step) so it's never fetched at runtime.
        "nltk",
    )
    # Bake the NLTK tokenizer data into the image. whisperx (run in an isolated uv env at
    # runtime) searches /usr/local/share/nltk_data, so downloading here once avoids the
    # flaky runtime download that failed with "Connection reset by peer".
    .run_commands(
        "python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab",
        # TRIBE's text pipeline uses spaCy's en_core_web_lg (~400 MB) to add context to
        # transcribed words. Bake it into the image so it isn't pip-downloaded into the
        # ephemeral container filesystem on every cold start.
        "python -m spacy download en_core_web_lg",
    )
    # Point heavy caches at the persistent Volume (/cache) and the baked NLTK data, so:
    #  - whisperx's ~1.2 GB alignment model downloads ONCE, then persists across runs
    #    (TORCH_HOME / HF_HOME), instead of re-downloading on every video.
    #  - the speech step finds punkt_tab without any network call (NLTK_DATA).
    # These env vars are inherited by the whisperx subprocess that TRIBE spawns. (uv's own
    # package cache is left at its default — Modal's mirror reinstalls it in <1s anyway.)
    .env({
        "HF_HOME": f"{CACHE_DIR}/hf",
        "TORCH_HOME": f"{CACHE_DIR}/torch",
        "NLTK_DATA": "/usr/local/share/nltk_data",
    })
    # Bundle our scoring code into the image.
    .add_local_python_source("networks", "scoring", "insights", "preprocess")
)

app = modal.App("braincomputer-tribe")

# Persistent caches: model weights + prebuilt atlas masks + reference stats.
cache_vol = modal.Volume.from_name("braincomputer-cache", create_if_missing=True)

# HF token for gated Llama 3.2. Create with:
#   modal secret create huggingface HF_TOKEN=hf_xxx
hf_secret = modal.Secret.from_name("huggingface")


@app.cls(
    image=image,
    gpu="L4",                       # 24 GB; bump to "A10G"/"A100" if VRAM is tight
    volumes={CACHE_DIR: cache_vol},
    secrets=[hf_secret],
    scaledown_window=120,           # keep warm 2 min between calls, then scale to zero
    # A full video runs all 4 extractors (whisperx speech + Llama text + W2v-BERT audio
    # + V-JEPA2-giant video) and predict. The first cold run also downloads ~2 GB of
    # weights, and V-JEPA2-giant encoding is heavy on an L4 (~10s/chunk). 600s was too
    # tight; 1800s gives the cold run headroom. Warm runs are far faster.
    timeout=1800,
)
class TribeWorker:
    @modal.enter()
    def load(self):
        """Load the model once per container. Cached weights make warm starts fast."""
        os.environ.setdefault("HF_HOME", f"{CACHE_DIR}/hf")
        os.environ.setdefault("NETWORKS_CACHE", f"{CACHE_DIR}/networks_fsaverage5.npz")
        os.environ.setdefault("REFERENCE_STATS", f"{CACHE_DIR}/reference_stats.json")
        # HF token from the secret (works for huggingface_hub auto-auth).
        if os.environ.get("HF_TOKEN"):
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

        from tribev2 import TribeModel

        self.model = TribeModel.from_pretrained(MODEL_ID, cache_folder=f"{CACHE_DIR}/tribe")
        # Warm the atlas masks (builds + caches to the Volume on first cold start).
        import networks
        networks.get_masks()
        cache_vol.commit()

    @modal.method()
    def analyze_bytes(self, data: bytes, filename: str, job_id: str = "") -> dict:
        """Run the full pipeline on raw file bytes and return a result dict."""
        import tempfile

        import numpy as np  # noqa: F401  (used indirectly via scoring)
        import preprocess
        import scoring
        import insights

        t0 = time.time()
        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            in_path = fh.name

        routed = None
        try:
            routed = preprocess.route(in_path)
            df = self.model.get_events_dataframe(**routed.kwargs)
            preds, _segments = self.model.predict(events=df)

            result = scoring.score_predictions(preds, duration_s=routed.duration_s)
            cards = insights.generate_insights(result, routed.modality, routed.experimental)

            return {
                "schema_version": "1.0",
                "job_id": job_id,
                "modality": routed.modality,
                "status": "done",
                "error": None,
                "engagement_index": result.engagement_index,
                "networks": result.networks,
                "timeline": result.timeline,
                "insights": cards,
                "meta": {
                    "experimental": routed.experimental,
                    "duration_s": routed.duration_s,
                    "n_vertices": result.n_vertices,
                    "n_timesteps": result.n_timesteps,
                    "model": MODEL_ID,
                    "processing_ms": int((time.time() - t0) * 1000),
                },
            }
        except Exception as exc:  # surface a structured error to the backend
            return {
                "schema_version": "1.0", "job_id": job_id,
                "modality": (routed.modality if routed else "video"),
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "engagement_index": 0, "networks": [], "timeline": {"t": [], "overall": []},
                "insights": [], "meta": {"model": MODEL_ID},
            }
        finally:
            for p in ([in_path] + (routed.cleanup if routed else [])):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            # Persist any newly downloaded caches (whisperx uv env, torch alignment
            # model, HF assets) to the Volume so later runs skip the re-download.
            try:
                cache_vol.commit()
            except Exception:
                pass


@app.local_entrypoint()
def main(input: str):
    """Local smoke test: `modal run inference/modal_app.py --input sample_reel.mp4`."""
    import json

    with open(input, "rb") as f:
        data = f.read()
    out = TribeWorker().analyze_bytes.remote(data, os.path.basename(input), job_id="local-test")
    print(json.dumps(out, indent=2))
