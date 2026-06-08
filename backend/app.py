"""BrainComputer Flask API.

Endpoints:
  POST /api/analyze     multipart 'file' -> {job_id, status}
  GET  /api/jobs/<id>   -> result JSON (shared/result_schema.json) or {status}
  GET  /healthz         -> {ok: true}

The backend does no GPU work: it stores the upload, hands it to the Modal worker
(inference_client), tracks a small JSON job record, and serves results. See
config.py for env-driven settings (R2, Modal, MOCK_INFERENCE, CORS).
"""

from __future__ import annotations

import json
import os
import time
import uuid

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import config
import inference_client
import storage

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
store = storage.get_store()

os.makedirs(config.JOBS_DIR, exist_ok=True)

# Map upload extension -> modality (mirrors inference/preprocess.py).
_EXT_MODALITY = {
    **{e: "video" for e in (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi")},
    **{e: "image" for e in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")},
    **{e: "audio" for e in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")},
    **{e: "text" for e in (".txt", ".md")},
}


# ---- job record helpers ----------------------------------------------------

def _job_path(job_id: str) -> str:
    return os.path.join(config.JOBS_DIR, f"{job_id}.json")


def _save_job(rec: dict) -> None:
    with open(_job_path(rec["job_id"]), "w") as f:
        json.dump(rec, f)


def _load_job(job_id: str) -> dict | None:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---- CORS ------------------------------------------------------------------

@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin", "")
    if origin in config.ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/analyze", methods=["OPTIONS"])
@app.route("/api/jobs/<job_id>", methods=["OPTIONS"])
def cors_preflight(job_id=None):
    return ("", 204)


# ---- routes ----------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "mock": config.MOCK_INFERENCE})


@app.post("/api/analyze")
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "no file provided (field 'file')"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_EXTS:
        return jsonify({"error": f"unsupported file type '{ext}'"}), 415
    modality = _EXT_MODALITY.get(ext, "video")

    data = f.read()
    if not data:
        return jsonify({"error": "empty file"}), 400

    job_id = uuid.uuid4().hex
    store.put_upload(f"{job_id}{ext}", data)

    try:
        call_id = inference_client.spawn(data, filename, job_id, modality)
    except Exception as exc:
        return jsonify({"error": f"failed to start inference: {exc}"}), 502

    rec = {
        "job_id": job_id, "filename": filename, "modality": modality,
        "ext": ext, "call_id": call_id, "status": "running",
        "created_at": time.time(),
    }
    _save_job(rec)
    return jsonify({"job_id": job_id, "status": "running", "modality": modality}), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    rec = _load_job(job_id)
    if rec is None:
        return jsonify({"error": "unknown job"}), 404

    # Already finished? Serve the stored result.
    if rec["status"] in ("done", "error"):
        result = store.get_result(job_id)
        if result is not None:
            return jsonify(result)

    # Poll the worker.
    try:
        result = inference_client.poll(rec.get("call_id"), job_id, rec["filename"], rec["modality"])
    except Exception as exc:
        rec["status"] = "error"
        _save_job(rec)
        return jsonify({"job_id": job_id, "status": "error", "error": str(exc)}), 200

    if result is None:
        return jsonify({"job_id": job_id, "status": "running"}), 200

    rec["status"] = result.get("status", "done")
    _save_job(rec)
    store.put_result(job_id, result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
