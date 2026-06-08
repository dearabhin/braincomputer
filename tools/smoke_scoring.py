"""Offline smoke test for the scoring + insights layer (no GPU / no model needed).

Builds a synthetic fsaverage5 network-mask cache, fabricates a plausible
(n_timesteps, n_vertices) prediction array with a visual/auditory bias, then runs
scoring + insights and validates the output against shared/result_schema.json.

Run: python3 tools/smoke_scoring.py
"""

import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "inference"))

# Point caches at a temp dir BEFORE importing networks/scoring.
tmp = tempfile.mkdtemp(prefix="bc_smoke_")
os.environ["NETWORKS_CACHE"] = os.path.join(tmp, "networks.npz")
os.environ["REFERENCE_STATS"] = os.path.join(tmp, "ref.json")

import networks  # noqa: E402
import scoring  # noqa: E402
import insights  # noqa: E402

N_VERTS = networks.N_VERTS
rng = np.random.default_rng(0)

# --- Build a synthetic mask cache: assign contiguous vertex blocks per network. ---
keys = networks.NETWORK_KEYS
masks = {}
block = N_VERTS // (len(keys) + 1)
for i, k in enumerate(keys):
    m = np.zeros(N_VERTS, dtype=bool)
    m[i * block:(i + 1) * block] = True
    masks[k] = m
networks.NetworkMasks(masks=masks, n_verts=N_VERTS).to_npz(os.environ["NETWORKS_CACHE"])
networks.get_masks.cache_clear()

# --- Fabricate predictions: 20 timesteps, visual+auditory strongly driven early. ---
T = 20
preds = rng.normal(0.15, 0.05, size=(T, N_VERTS)).astype(np.float32)
vis = masks["visual"]
aud = masks["auditory"]
preds[:6, vis] += 0.9          # strong early visual hook
preds[:, aud] += 0.4           # steady audio
preds[:, masks["language"]] -= 0.05  # weak language -> should trigger a caption tip

result = scoring.score_predictions(preds, duration_s=float(T))
cards = insights.generate_insights(result, modality="video", experimental=False)

out = {
    "schema_version": "1.0", "job_id": "smoke", "modality": "video", "status": "done",
    "error": None, "engagement_index": result.engagement_index,
    "networks": result.networks, "timeline": result.timeline, "insights": cards,
    "meta": {"experimental": False, "duration_s": float(T), "n_vertices": result.n_vertices,
             "n_timesteps": result.n_timesteps, "model": "synthetic", "processing_ms": 0},
}

print(json.dumps({"engagement_index": out["engagement_index"],
                  "networks": {n["key"]: n["score"] for n in out["networks"]},
                  "n_insights": len(cards),
                  "insight_titles": [c["title"] for c in cards],
                  "timeline_len": len(out["timeline"]["overall"])}, indent=2))

# --- Validate against the JSON schema if jsonschema is available. ---
try:
    import jsonschema
    with open(os.path.join(ROOT, "shared", "result_schema.json")) as f:
        schema = json.load(f)
    jsonschema.validate(out, schema)
    print("\nSCHEMA: valid ✓")
except ImportError:
    # Minimal manual checks.
    assert 0 <= out["engagement_index"] <= 100
    assert len(out["networks"]) == len(keys)
    assert all(0 <= n["score"] <= 100 for n in out["networks"])
    assert len(out["timeline"]["t"]) == len(out["timeline"]["overall"])
    print("\nSCHEMA: jsonschema not installed; manual checks passed ✓")

print("SMOKE OK")
