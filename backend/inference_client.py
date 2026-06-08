"""Bridge from the Flask backend to the Modal GPU worker.

Spawns the deployed Modal worker asynchronously and polls for its result, so the
HTTP request returns immediately with a job id. In MOCK_INFERENCE mode it returns a
deterministic canned result instead -- no Modal account or GPU required, ideal for
frontend development.
"""

from __future__ import annotations

import hashlib

import config


# ---- Mock mode -------------------------------------------------------------

def _mock_result(job_id: str, filename: str, modality: str) -> dict:
    """Deterministic fake result derived from the filename hash (stable per file)."""
    h = int(hashlib.sha256(filename.encode()).hexdigest(), 16)

    def pick(lo, hi, salt):
        return round(lo + ((h >> salt) % 1000) / 1000 * (hi - lo), 1)

    nets = [
        ("visual", "Visual", pick(40, 90, 0)),
        ("auditory", "Auditory", pick(30, 85, 8)),
        ("language", "Language / Semantic", pick(20, 80, 16)),
        ("emotional_social", "Emotional & Social", pick(25, 80, 24)),
        ("default_mode", "Default-Mode (Narrative)", pick(30, 70, 32)),
        ("multisensory", "Multisensory Integration", pick(25, 70, 40)),
    ]
    networks = [{"key": k, "label": lab, "score": s, "description": ""} for k, lab, s in nets]
    idx = round(sum(s for _, _, s in nets) / len(nets), 1)
    n = 12
    t = list(range(n))
    overall = [round(idx + 12 * (0.5 - ((h >> i) % 100) / 100), 1) for i in range(n)]
    return {
        "schema_version": "1.0", "job_id": job_id, "modality": modality, "status": "done",
        "error": None, "engagement_index": idx, "networks": networks,
        "timeline": {"t": t, "overall": overall, "by_network": {}},
        "insights": [
            {"severity": "tip", "title": "Mock result",
             "body": "MOCK_INFERENCE is on — these numbers are fabricated for UI development."},
        ],
        "meta": {"experimental": modality in ("image", "text"), "duration_s": float(n),
                 "n_vertices": 20484, "n_timesteps": n, "model": "mock", "processing_ms": 5},
    }


# ---- Real Modal mode -------------------------------------------------------

def spawn(data: bytes, filename: str, job_id: str, modality: str) -> str | None:
    """Start inference. Returns a Modal call id, or None in mock mode."""
    if config.MOCK_INFERENCE:
        return None
    import modal
    worker = modal.Cls.from_name(config.MODAL_APP_NAME, config.MODAL_CLS_NAME)()
    method = getattr(worker, config.MODAL_METHOD)
    call = method.spawn(data, filename, job_id)
    return call.object_id


def poll(call_id: str | None, job_id: str, filename: str, modality: str) -> dict | None:
    """Return the result if ready, else None (still running). Mock returns instantly."""
    if config.MOCK_INFERENCE:
        return _mock_result(job_id, filename, modality)
    import modal
    fc = modal.functions.FunctionCall.from_id(call_id)
    try:
        return fc.get(timeout=0)
    except TimeoutError:
        return None
