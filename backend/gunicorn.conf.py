"""Gunicorn config for the BrainComputer API on a small CPU droplet.

The backend is I/O-bound (object storage + polling Modal), so a handful of gthread
workers is plenty for a portfolio-scale load. No GPU here.
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
worker_class = "gthread"
threads = int(os.environ.get("THREADS", "4"))
timeout = 120
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
