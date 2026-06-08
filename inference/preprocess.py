"""Modality detection and routing for TRIBE v2 inference.

TRIBE v2's ``get_events_dataframe`` accepts ``video_path``, ``audio_path`` or
``text_path`` natively. We support four creator inputs:

  - video  -> video_path (frames + audio handled by the library)
  - audio  -> audio_path
  - text   -> text_path (library does TTS + word-level timing internally)
  - image  -> NOT native; we synthesize a short static clip with ffmpeg and feed it
              as video_path. This mirrors the paper's flashed-image localizer
              (Fig. 4), where static images evoke a measurable visual response.

``text`` and ``image`` are degenerate cases for a temporal video+audio+text model,
so they are flagged ``experimental`` and surfaced as such in the UI.
"""

from __future__ import annotations

import mimetypes
import os
import subprocess
import tempfile
from dataclasses import dataclass

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
TEXT_EXTS = {".txt", ".md"}

# Duration of the synthesized clip for image inputs.
IMAGE_CLIP_SECONDS = 2.0
EXPERIMENTAL_MODALITIES = {"image", "text"}


@dataclass
class RoutedInput:
    modality: str            # video | image | audio | text
    kwargs: dict             # passed straight to get_events_dataframe(**kwargs)
    experimental: bool
    cleanup: list[str]       # temp paths to remove after inference
    duration_s: float | None


def detect_modality(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in TEXT_EXTS:
        return "text"
    # Fall back to mimetype sniffing.
    mime, _ = mimetypes.guess_type(path)
    if mime:
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("text/"):
            return "text"
    raise ValueError(f"unsupported file type: {os.path.basename(path)} ({ext or mime})")


def image_to_clip(image_path: str, seconds: float = IMAGE_CLIP_SECONDS) -> str:
    """Render a static image into a short H.264 mp4 (silent) via ffmpeg."""
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", image_path,
        "-t", str(seconds),
        "-vf", "scale='min(720,iw)':-2,format=yuv420p,fps=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out,
    ]
    subprocess.run(cmd, check=True)
    return out


def probe_duration(path: str) -> float | None:
    """Best-effort media duration in seconds via ffprobe; None if unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def route(path: str, modality: str | None = None) -> RoutedInput:
    """Map an input file to get_events_dataframe kwargs."""
    modality = modality or detect_modality(path)
    cleanup: list[str] = []

    if modality == "video":
        return RoutedInput("video", {"video_path": path}, False, cleanup, probe_duration(path))

    if modality == "audio":
        return RoutedInput("audio", {"audio_path": path}, False, cleanup, probe_duration(path))

    if modality == "text":
        return RoutedInput("text", {"text_path": path}, True, cleanup, None)

    if modality == "image":
        clip = image_to_clip(path)
        cleanup.append(clip)
        return RoutedInput("image", {"video_path": clip}, True, cleanup, IMAGE_CLIP_SECONDS)

    raise ValueError(f"unsupported modality: {modality}")
