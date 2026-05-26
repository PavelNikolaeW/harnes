"""Entry point for the harnes agent runtime.

В v0 этот скрипт проверяет:
- что конфиг грузится;
- что трассировка инициализирована;
- что endpoint'ы достижимы (smoke-check будет добавлен в задаче #13).

Полная wire-up метацикла придёт в задаче #9.
"""
from __future__ import annotations

import logging
import sys

import structlog

from harnes.config import get_settings
from harnes.llm import health_check
from harnes.telemetry import setup_logging


def main() -> int:
    settings = get_settings()
    setup_logging(settings.logging.level)
    log = structlog.get_logger()

    log.info(
        "harnes.boot",
        version="0.0.1",
        llm_endpoint=settings.llm.api_base,
        llm_model=settings.llm.model,
        memory={
            "lancedb": str(settings.memory.lancedb_path),
            "qdrant": settings.memory.qdrant_url,
            "neo4j": settings.memory.neo4j_uri,
        },
    )

    # LLM connectivity smoke-check (task #2).
    if not health_check():
        log.error("harnes.boot.llm_unreachable")
        return 1

    # TODO(#9): wire up meta-cycle tick driver
    log.warning("metacycle.not_yet_wired", next_task="#9")

    return 0


if __name__ == "__main__":
    sys.exit(main())
