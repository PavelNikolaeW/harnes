"""Embeddings client.

Primary: LiteLLM `embedding()` → отправляет в `/v1/embeddings` на ll-router.
Fallback: fastembed + `paraphrase-multilingual-mpnet-base-v2` локально на CPU.

Переключается по конфигу `embeddings.use_server`:
- True — пробуем сервер, при любой ошибке (404, timeout, connection) пишем
  warning и автоматически делаем fallback на fastembed. После N подряд
  фейлов помечаем сервер как broken и не дёргаем `broken_ttl_s` секунд —
  чтобы не висеть на 60s LiteLLM-таймауте на каждый embed.
- False — сразу fastembed.

Это позволяет включить `use_server: true` заранее (документ
`docs/router_roadmap.md` R1) — агент будет работать через fastembed, пока
endpoint не появится, и сам переключится на сервер когда он заработает.

Public API:
- embed(texts) -> list[list[float]]
- reset_server_state()  — тестовая утилита, сбрасывает broken-cache

См. `agent_architecture.html` § 17, `docs/router_roadmap.md` R1.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import structlog
from litellm import embedding

from harnes.config import get_settings

log = structlog.get_logger()


# ---------- Server-state cache ----------
#
# Если сервер ответил ошибкой — не дёргаем его _broken_ttl_s секунд, чтобы не
# страдать от 60s LiteLLM-таймаута на каждый embed. Простой in-memory маркер,
# сбрасывается при reset_server_state() или по TTL.
_broken_until: float = 0.0
_broken_ttl_s: float = 60.0


def reset_server_state() -> None:
    """Сбросить кеш «сервер сломан». Используется в тестах."""
    global _broken_until
    _broken_until = 0.0


@lru_cache(maxsize=1)
def _fastembed_model() -> Any:
    """Lazy-load fastembed модели. Первый вызов скачивает модель.

    Если в конфиге запрошен `BAAI/bge-m3` (не в curated-списке fastembed) —
    автоматический фолбэк на `intfloat/multilingual-e5-large` (1024-dim,
    multilingual, эквивалентный по качеству для русского/английского).
    """
    from fastembed import TextEmbedding

    settings = get_settings()
    requested = settings.embeddings.model
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    # Fallback chain — стабильно подгружаемые модели в порядке убывания качества.
    # multilingual-e5-large сейчас глючит из-за external-data файла; mpnet и
    # bge-large более надёжны в текущей версии fastembed.
    fallback_chain = [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # multilingual, 768d
        "BAAI/bge-large-en-v1.5",  # english only, 1024d
        "BAAI/bge-small-en-v1.5",  # last resort, 384d
    ]
    if requested in supported:
        model_name = requested
    else:
        model_name = next((m for m in fallback_chain if m in supported), fallback_chain[-1])
        log.warning(
            "embeddings.fastembed.fallback",
            requested=requested,
            fallback=model_name,
            reason="not in fastembed supported list",
        )
    log.info("embeddings.fastembed.loading", model=model_name)
    model = TextEmbedding(model_name=model_name)
    log.info("embeddings.fastembed.loaded", model=model_name)
    return model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed списка текстов в векторы.

    Маршрутизация по `settings.embeddings.use_server`:
    - True:  пробуем серверный endpoint через LiteLLM; при ошибке —
      graceful fallback на fastembed + broken-state кеш на 60s.
    - False: сразу fastembed на CPU локально.
    """
    if not texts:
        return []

    settings = get_settings()
    if not settings.embeddings.use_server:
        return _embed_via_fastembed(texts)

    # use_server=True — но сервер недавно отвечал ошибкой? Пропустить, не дёргать.
    global _broken_until
    if _broken_until > time.monotonic():
        return _embed_via_fastembed(texts)

    try:
        return _embed_via_server(texts)
    except Exception as exc:  # noqa: BLE001 — любая ошибка → fallback
        _broken_until = time.monotonic() + _broken_ttl_s
        log.warning(
            "embeddings.server.failed",
            error=str(exc),
            error_type=type(exc).__name__,
            fallback="fastembed",
            broken_for_seconds=_broken_ttl_s,
        )
        return _embed_via_fastembed(texts)


def _embed_via_server(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    log.debug(
        "embeddings.server.start",
        count=len(texts),
        model=settings.embeddings.model,
    )
    response = embedding(
        model=f"openai/{settings.embeddings.model}",
        input=texts,
        api_base=settings.llm.api_base,
        api_key=settings.llm.api_key,
        timeout=settings.llm.timeout,
    )
    vectors: list[list[float]] = [item["embedding"] for item in response.data]
    log.debug("embeddings.server.done", count=len(vectors))
    return vectors


def _embed_via_fastembed(texts: list[str]) -> list[list[float]]:
    model = _fastembed_model()
    log.debug("embeddings.fastembed.start", count=len(texts))
    vectors: list[list[float]] = [vec.tolist() for vec in model.embed(texts)]
    log.debug("embeddings.fastembed.done", count=len(vectors))
    return vectors
