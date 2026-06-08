"""Turn TRIBE v2's predicted brain activity into creator-facing scores.

Input is the raw prediction from ``TribeModel.predict`` -- an
``(n_timesteps, n_vertices)`` array on fsaverage5. We:

1. Aggregate activation magnitude within each functional network (networks.py).
2. Normalize each network against a bundled reference distribution of "typical"
   content so the 0-100 scores are comparable across uploads rather than arbitrary.
3. Blend the network scores into a composite Neural Engagement Index using the
   documented, transparent weights in ``networks.NETWORK_META``.
4. Produce a per-second engagement timeline (overall + per network) for pacing
   insights.

IMPORTANT: these are *proxies derived from predicted brain activity*, not measured
engagement and not a guarantee of reach. The framing lives in the UI; this module
just produces numbers honestly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from networks import NETWORK_KEYS, NETWORK_META, get_masks

# fMRI TR for the reference timeline resampling (TRIBE predicts at ~1 frame / 1.49s
# for HCP-like data; we resample the timeline to 1 Hz for display).
REFERENCE_STATS = os.environ.get("REFERENCE_STATS", "/cache/reference_stats.json")

# Fallback reference distribution (per-network mean activation magnitude + std),
# used until a real reference set is computed via build_reference(). These are
# deliberately conservative placeholders; replace with build_reference() output.
_DEFAULT_REF = {
    "visual": [0.32, 0.12],
    "auditory": [0.24, 0.10],
    "language": [0.21, 0.09],
    "emotional_social": [0.18, 0.08],
    "default_mode": [0.16, 0.07],
    "multisensory": [0.19, 0.08],
}


@dataclass
class ScoreResult:
    engagement_index: float
    networks: list[dict]
    timeline: dict
    n_timesteps: int
    n_vertices: int


def _load_reference() -> dict[str, tuple[float, float]]:
    if os.path.exists(REFERENCE_STATS):
        with open(REFERENCE_STATS) as f:
            raw = json.load(f)
    else:
        raw = _DEFAULT_REF
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def _network_activation(preds: np.ndarray) -> dict[str, np.ndarray]:
    """Mean absolute activation per network, per timestep -> (n_timesteps,) each.

    TRIBE predictions are z-scored-ish encoded responses; magnitude (|activation|)
    captures how strongly a network is driven regardless of sign.
    """
    masks = get_masks()
    mag = np.abs(np.asarray(preds, dtype=np.float32))  # (T, V)
    out: dict[str, np.ndarray] = {}
    for key in NETWORK_KEYS:
        m = masks.vertices(key)
        if m.shape[0] != mag.shape[1]:
            # Defensive: align mask length to actual vertex count.
            m = _resize_mask(m, mag.shape[1])
        out[key] = mag[:, m].mean(axis=1) if m.any() else np.zeros(mag.shape[0], np.float32)
    return out


def _resize_mask(mask: np.ndarray, n: int) -> np.ndarray:
    if mask.shape[0] == n:
        return mask
    out = np.zeros(n, dtype=bool)
    k = min(mask.shape[0], n)
    out[:k] = mask[:k]
    return out


def _to_score(value: float, mean: float, std: float) -> float:
    """Map an activation level to 0-100 via a logistic on its z-score.

    z = (value - mean) / std; score = 100 * sigmoid(z). This keeps typical content
    near 50, strong content toward 100, weak toward 0, without hard clipping.
    """
    std = max(std, 1e-6)
    z = (value - mean) / std
    return float(100.0 / (1.0 + np.exp(-z)))


def _resample_to_1hz(series: np.ndarray, duration_s: float) -> tuple[list[float], list[float]]:
    """Resample a per-timestep series to ~1 Hz timestamps for display."""
    n = len(series)
    if n == 0 or duration_s <= 0:
        return [], []
    src_t = np.linspace(0, duration_s, n)
    n_out = max(2, int(round(duration_s)))
    dst_t = np.linspace(0, duration_s, n_out)
    dst_v = np.interp(dst_t, src_t, series)
    return [round(float(t), 2) for t in dst_t], [round(float(v), 4) for v in dst_v]


def score_predictions(preds: np.ndarray, duration_s: float | None = None) -> ScoreResult:
    """Compute network scores, composite index, and timeline from raw predictions."""
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2:
        raise ValueError(f"expected (n_timesteps, n_vertices), got shape {preds.shape}")
    n_timesteps, n_vertices = preds.shape
    if duration_s is None:
        duration_s = float(n_timesteps)  # assume ~1s/step if unknown

    ref = _load_reference()
    per_ts = _network_activation(preds)  # key -> (T,)

    networks_out: list[dict] = []
    timeline_by_net: dict[str, list[float]] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for key in NETWORK_KEYS:
        meta = NETWORK_META[key]
        mean, std = ref.get(key, (0.2, 0.1))
        ts = per_ts[key]
        agg = float(ts.mean()) if ts.size else 0.0
        score = round(_to_score(agg, mean, std), 1)
        networks_out.append({
            "key": key,
            "label": meta["label"],
            "score": score,
            "description": meta["description"],
        })
        weighted_sum += score * meta["weight"]
        total_weight += meta["weight"]

        # Per-network timeline (scored per timestep, then resampled).
        scored_ts = np.array([_to_score(v, mean, std) for v in ts], dtype=np.float32)
        _, net_series = _resample_to_1hz(scored_ts, duration_s)
        timeline_by_net[key] = net_series

    engagement_index = round(weighted_sum / total_weight, 1) if total_weight else 0.0

    # Overall timeline = weighted mean of network timelines per timestep.
    overall_per_ts = np.zeros(n_timesteps, dtype=np.float32)
    for key in NETWORK_KEYS:
        mean, std = ref.get(key, (0.2, 0.1))
        scored = np.array([_to_score(v, mean, std) for v in per_ts[key]], dtype=np.float32)
        overall_per_ts += scored * NETWORK_META[key]["weight"]
    overall_per_ts /= total_weight
    t_axis, overall_series = _resample_to_1hz(overall_per_ts, duration_s)

    timeline = {"t": t_axis, "overall": overall_series, "by_network": timeline_by_net}

    return ScoreResult(
        engagement_index=engagement_index,
        networks=networks_out,
        timeline=timeline,
        n_timesteps=n_timesteps,
        n_vertices=n_vertices,
    )


def build_reference(prediction_arrays: list[np.ndarray], out_path: str = REFERENCE_STATS) -> dict:
    """Compute per-network mean/std over a set of reference clips and persist it.

    Run this once over a handful of 'typical' creator clips so 0-100 scores are
    calibrated. Each item is a raw (T, V) prediction array.
    """
    accum: dict[str, list[float]] = {k: [] for k in NETWORK_KEYS}
    for preds in prediction_arrays:
        per_ts = _network_activation(preds)
        for key in NETWORK_KEYS:
            accum[key].append(float(per_ts[key].mean()))
    stats = {k: [float(np.mean(v)), float(np.std(v) or 0.1)] for k, v in accum.items()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats
