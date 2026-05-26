"""World model store — temporal KG on Graphiti + Neo4j.

См. `agent_architecture.html` § 14.

Graphiti — async-first. Этот модуль оборачивает его в sync-API для нашего
синхронного метацикла через asyncio.run(). Это создаёт+уничтожает event-loop
на каждый вызов; для v0.2 этого достаточно. В будущем — persistent async-thread.

LLM-клиент Graphiti указан на ll-router (OpenAIClient + base_url).
Embedder — наш fastembed (BGE-M3) обёрнутый в EmbedderClient-адаптер,
потому что у ll-router нет /v1/embeddings.
"""
from __future__ import annotations

import asyncio
import re
import socket
from datetime import UTC, datetime
from typing import Any, Iterable

import structlog

from harnes.config import get_settings

log = structlog.get_logger()


# ---------- Embedder adapter ----------


def _get_embedder_client():
    """Lazy импорт Graphiti'ных типов + конструкция fastembed-адаптера."""
    from graphiti_core.embedder.client import EmbedderClient

    from harnes.llm.embeddings import embed as harnes_embed

    class FastEmbedAdapter(EmbedderClient):
        """Graphiti EmbedderClient над harnes.llm.embeddings.embed (BGE-M3)."""

        async def create(
            self,
            input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
        ) -> list[float]:
            # Token-id paths не поддерживаем — fastembed работает на тексте.
            if isinstance(input_data, str):
                vectors = harnes_embed([input_data])
                return vectors[0] if vectors else []
            if isinstance(input_data, list) and (
                not input_data or isinstance(input_data[0], str)
            ):
                vectors = harnes_embed(list(input_data))  # type: ignore[arg-type]
                return vectors[0] if vectors else []
            raise NotImplementedError(
                "FastEmbedAdapter: token-ids input not supported"
            )

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            return harnes_embed(input_data_list)

    return FastEmbedAdapter()


# ---------- LLM client for Graphiti ----------


def _get_llm_client():
    """Graphiti LLM client указан на ll-router.

    Используем OpenAIGenericClient (chat.completions API), а не OpenAIClient —
    последний делает client.responses.parse(), а ll-router этот endpoint
    не реализует.
    """
    from graphiti_core.llm_client import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    settings = get_settings()
    llm_config = LLMConfig(
        api_key=settings.llm.api_key,
        base_url=settings.llm.api_base,
        model=settings.llm.model,
        small_model=settings.llm.tiers.get("light", settings.llm.model),
    )
    return OpenAIGenericClient(config=llm_config)


# ---------- No-op cross-encoder ----------


def _get_cross_encoder():
    """No-op reranker: возвращает passages с равными score'ами без LLM-вызова.

    Graphiti по умолчанию создаёт OpenAIRerankerClient, который требует
    OPENAI_API_KEY env. Нам это не подходит — ll-router OpenAI-совместимый,
    но reranker'ом в Graphiti не пользуемся. Поэтому identity-ranking.
    """
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class NoopReranker(CrossEncoderClient):
        async def rank(
            self, query: str, passages: list[str]
        ) -> list[tuple[str, float]]:
            # Identity: каждое passage получает score=1.0, порядок сохраняется.
            return [(p, 1.0) for p in passages]

    return NoopReranker()


# ---------- WorldModelStore ----------


class WorldModelStore:
    """Реальная Graphiti-обёртка над Neo4j.

    Sync API поверх async Graphiti. Используем **persistent event loop** —
    asyncio.run() пересоздавал бы loop на каждый вызов, ломая Neo4j driver
    Graphiti'я с "Event loop is closed". loop.run_until_complete на одном
    long-lived loop'е этого избегает.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ) -> None:
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self._graphiti: Any = None
        self._initialised = False
        # Если init упал — больше не пытаемся. Это критично, потому что Neo4j
        # driver по умолчанию retries долго (15s+ per call), и без кэширования
        # broken-state каждый тик будет блокироваться.
        self._broken = False
        # Persistent event loop — переживает между вызовами.
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro: Any) -> Any:
        return self._get_loop().run_until_complete(coro)

    def _bolt_reachable(self, timeout: float = 2.0) -> bool:
        """Быстрый TCP ping bolt-порта. Без этого Neo4j driver retries 15s+."""
        m = re.match(r"bolt://([^:/]+):?(\d+)?", self.neo4j_uri)
        if not m:
            return False
        host = m.group(1)
        port = int(m.group(2) or "7687")
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (OSError, socket.gaierror):
            return False

    def _get_graphiti(self) -> Any:
        """Lazy-создаёт Graphiti instance. Возвращает None если broken."""
        if self._broken:
            return None
        if self._graphiti is not None and self._initialised:
            return self._graphiti

        # Pre-check TCP — экономит до 15s timeout на каждом вызове.
        if not self._bolt_reachable():
            log.warning("world.neo4j.unreachable", uri=self.neo4j_uri)
            self._broken = True
            return None

        from graphiti_core import Graphiti

        if self._graphiti is None:
            self._graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                llm_client=_get_llm_client(),
                embedder=_get_embedder_client(),
                cross_encoder=_get_cross_encoder(),
            )

        try:
            self._run(self._graphiti.build_indices_and_constraints())
            self._initialised = True
            log.info("world.graphiti.initialised", uri=self.neo4j_uri)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "world.graphiti.init_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            self._broken = True
            return None
        return self._graphiti

    # ---------- public API ----------

    def add_episode(
        self,
        name: str,
        episode_body: str,
        reference_time: datetime | None = None,
        source_description: str = "harnes",
    ) -> None:
        """Запись эпизода в KG. Graphiti сам извлекает entities/relations
        через LLM и пишет в Neo4j."""
        try:
            from graphiti_core.nodes import EpisodeType

            if reference_time is None:
                reference_time = datetime.now(UTC)
            g = self._get_graphiti()
            if g is None:
                log.debug("world.add_episode.skipped_broken", name=name)
                return
            self._run(
                g.add_episode(
                    name=name,
                    episode_body=episode_body,
                    reference_time=reference_time,
                    source=EpisodeType.text,
                    source_description=source_description,
                )
            )
            log.debug(
                "world.add_episode.done",
                name=name,
                length=len(episode_body),
            )
        except Exception as exc:  # noqa: BLE001
            # Не блокируем метацикл — world_update это side-channel.
            log.warning(
                "world.add_episode.failed",
                name=name,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def search(
        self,
        query: str,
        k: int = 10,
        valid_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск по KG. Возвращает список dict'ов с полями id/labels/properties
        для совместимости с MemoryRouter._recall_world."""
        try:
            g = self._get_graphiti()
            if g is None:
                return []
            results = self._run(g.search(query=query, num_results=k))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "world.search.failed",
                query=query[:100],
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

        out: list[dict[str, Any]] = []
        for r in results:
            # Graphiti возвращает разные типы (EntityEdge, EntityNode, etc.) —
            # сводим к универсальной форме.
            item: dict[str, Any] = {
                "id": str(getattr(r, "uuid", getattr(r, "id", ""))),
                "labels": [type(r).__name__],
                "properties": {},
            }
            for attr in ("name", "fact", "summary", "created_at", "valid_at", "invalid_at"):
                v = getattr(r, attr, None)
                if v is not None:
                    item["properties"][attr] = str(v) if not isinstance(v, (str, int, float, bool)) else v
            out.append(item)
        return out

    def close(self) -> None:
        """Закрыть соединение с Neo4j и event loop."""
        if self._graphiti is not None:
            try:
                self._run(self._graphiti.close())
            except Exception as exc:  # noqa: BLE001
                log.warning("world.close.failed", error=str(exc))
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
            self._loop = None
