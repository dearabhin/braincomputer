"""Backend configuration, all environment-driven.

The backend is pure orchestration (no GPU): accept an upload, stash it in object
storage, hand it to the Modal GPU worker, and serve the result back. Two modes:

  - MOCK_INFERENCE=1  -> skip Modal entirely and return a canned result. Lets you
    develop/test the frontend with zero GPU cost.
  - default            -> spawn the deployed Modal worker and poll for the result.
"""

import os


def _bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


# --- Inference ---
MOCK_INFERENCE = _bool("MOCK_INFERENCE")
MODAL_APP_NAME = os.environ.get("MODAL_APP_NAME", "braincomputer-tribe")
MODAL_CLS_NAME = os.environ.get("MODAL_CLS_NAME", "TribeWorker")
MODAL_METHOD = os.environ.get("MODAL_METHOD", "analyze_bytes")

# --- Uploads ---
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
ALLOWED_EXTS = {
    ".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi",          # video
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",          # image
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",           # audio
    ".txt", ".md",                                             # text
}

# --- Storage (Cloudflare R2, S3-compatible). If unset, falls back to local disk. ---
R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "")             # https://<acct>.r2.cloudflarestorage.com
R2_BUCKET = os.environ.get("R2_BUCKET", "braincomputer")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
LOCAL_STORAGE_DIR = os.environ.get("LOCAL_STORAGE_DIR", os.path.join(os.path.dirname(__file__), ".data"))

# --- Job store (simple JSON files; swap for Redis/DB if scale demands) ---
JOBS_DIR = os.environ.get("JOBS_DIR", os.path.join(LOCAL_STORAGE_DIR, "jobs"))

# --- CORS: the Cloudflare Pages origin(s) allowed to call this API ---
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "https://braincomputer.in,https://www.braincomputer.in,http://localhost:8788",
    ).split(",") if o.strip()
]
