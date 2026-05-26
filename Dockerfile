# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# System deps (git for graphiti / skill bundles; build-essential for any C-ext deps).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv (https://docs.astral.sh/uv/)
RUN pip install --no-cache-dir uv

WORKDIR /app

# 1. Install dependencies first (layer caching).
#    Wildcard on uv.lock allows missing lock on first build.
COPY pyproject.toml /app/
COPY uv.lock* /app/
RUN uv sync --no-install-project

# 2. Copy source and re-sync to install the project itself.
COPY src/     /app/src/
COPY config/  /app/config/
COPY scripts/ /app/scripts/

RUN uv sync

# Data dirs (volume-mounted at runtime).
RUN mkdir -p /app/data/lancedb /app/data/traces /app/skills

CMD ["uv", "run", "python", "scripts/run_agent.py"]
