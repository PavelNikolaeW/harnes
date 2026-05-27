# syntax=docker/dockerfile:1.6
# WebUI image — отдельный сервис для admin-консоли.
# Тот же harnes-пакет, но с extras [webui]; нет heavy deps агента (LiteLLM/Graphiti/datasets).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Зависимости — со всеми extras [webui] (FastAPI, Jinja, SSE).
COPY pyproject.toml /app/
COPY uv.lock* /app/
RUN uv sync --no-install-project --extra webui

# Исходники.
COPY src/     /app/src/
COPY config/  /app/config/

RUN uv sync --extra webui

# Read-only volumes монтируются compose'ом: data/, skills/, config/.
EXPOSE 8000

# Production: один uvicorn-воркер (UI не нагружается).
CMD ["uv", "run", "uvicorn", "harnes.webui.app:app", "--host", "0.0.0.0", "--port", "8000"]
