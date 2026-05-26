"""Embeddings client.

Primary: LiteLLM `embedding()` → отправляет в `/v1/embeddings` на ll-router.
Fallback: fastembed + `BAAI/bge-m3` локально на CPU.

Переключается по конфигу `embeddings.use_server` (по умолчанию False, пока
ll-router не научится embeddings).

Public API:
- embed(texts) -> list[list[float]]

См. `agent_architecture.html` § 17.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog
from litellm import embedding

from harnes.config import get_settings

log = structlog.get_logger()


@lru_cache(maxsize=1)
def _fastembed_model() -> Any:
    """Lazy-load fastembed модели. Первый вызов скачивает ~500MB."""
    from fastembed import TextEmbedding

    settings = get_settings()
    model_name = settings.embeddings.model
    log.info("embeddings.fastembed.loading", model=model_name)
    model = TextEmbedding(model_name=model_name)
    log.info("embeddings.fastembed.loaded", model=model_name)
    return model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed списка текстов в векторы.

    Маршрутизация по `settings.embeddings.use_server`:
    - True:  серверный endpoint через LiteLLM
    - False: fastembed на CPU локально
    """
    if not texts:
        return []

    settings = get_settings()
    if settings.embeddings.use_server:
        return _embed_via_server(texts)
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
