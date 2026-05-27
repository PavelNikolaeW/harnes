"""harnes.webui — admin-консоль для исследовательского агента.

Отдельный сервис: читает persistent state агента (SQLite + LanceDB + Neo4j +
Qdrant) и предоставляет HTML-консоль для оператора. См. `webui/README.md`.

Запуск:
    docker compose up -d webui      # рекомендуется
    uv run python -m harnes.webui   # локально

Точки входа: `harnes.webui.__main__` (uvicorn dev-server) и `harnes.webui.app`
(ASGI application для production-сервера).
"""
from __future__ import annotations

from harnes.webui.app import create_app

__all__ = ["create_app"]
