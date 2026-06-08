# BrainComputer API — CPU-only orchestration service for a small droplet/VM.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
# Modal auth (token) and R2 creds are injected as env vars at runtime.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
