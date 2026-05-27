"""Unified recall API.

См. `agent_architecture.html` § 13.

MemoryRouter — тонкий координатор поверх трёх бэкендов:
- EpisodicStore (LanceDB)
- SemanticStore (Qdrant)
- WorldModelStore (Graphiti + Neo4j)

recall(query, types, k, filters, time_range) → MemoryBundle:
- Используется и в meta-`recall` (грубый запрос с низким k),
- и внутри ReAct через retrieval-tool (узкий запрос с высоким k).

В v0 эмбеддинг для семантического поиска считается лениво через
harnes.llm.embeddings.embed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import structlog

from harnes.memory.episodic import EpisodicStore, extract_terms
from harnes.memory.schema import (
    EpisodicRecord,
    MemoryBundle,
    MemoryType,
    SemanticRecord,
    WorldNode,
)
from harnes.memory.semantic import SemanticStore
from harnes.memory.world import WorldModelStore

log = structlog.get_logger()


class MemoryRouter:
    """Единый API для recall + write."""

    def __init__(
        self,
        episodic: EpisodicStore | None = None,
        semantic: SemanticStore | None = None,
        world: WorldModelStore | None = None,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.world = world

    def recall(
        self,
        query: str,
        types: list[MemoryType] | None = None,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> MemoryBundle:
        """Главный entry-point. None в types = всё, что есть."""
        if types is None:
            types = list(MemoryType)

        bundle = MemoryBundle()

        if MemoryType.EPISODIC in types and self.episodic is not None:
            bundle.episodic = self._recall_episodic(query, k, filters)

        if MemoryType.SEMANTIC in types and self.semantic is not None:
            bundle.semantic = self._recall_semantic(query, k, filters)

        if MemoryType.WORLD in types and self.world is not None:
            bundle.world = self._recall_world(query, k, time_range)

        # PROCEDURAL — подсказки скиллов; в v0 их собирает goal_arbitration
        # отдельно через SkillRegistry. Здесь можно добавить позже.

        return bundle

    # ---------- per-backend recall ----------

    def _recall_episodic(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None,
    ) -> list[EpisodicRecord]:
        """Keyword scoring по terms из query + recency-fallback.

        Прежняя реализация возвращала recent_steps игнорируя query — на длинном
        прогоне свежие шаги затирали трейс с ответом, и LLM не находил нужный
        контекст. ANN/embeddings — follow-up.
        """
        import json as _json

        assert self.episodic is not None
        terms = extract_terms(query)
        rows: list[dict[str, Any]] = []
        if terms:
            rows = self.episodic.search_steps_by_terms(terms=terms, limit=k)
        if not rows:
            rows = self.episodic.recent_steps(limit=k)
        return [
            EpisodicRecord(
                trajectory_id=UUID(r["trajectory_id"]),
                step_id=UUID(r["id"]),
                goal_id=UUID(r["goal_id"]),
                timestamp=r["timestamp"],
                step_type=r["step_type"],
                content=_json.loads(r["content_json"]),
            )
            for r in rows
        ]

    def _recall_semantic(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None,
    ) -> list[SemanticRecord]:
        assert self.semantic is not None
        # Ленивый импорт, чтобы тесты не дёргали fastembed на загрузке модуля.
        from harnes.llm.embeddings import embed

        vectors = embed([query])
        if not vectors:
            return []

        hits = self.semantic.search(vectors[0], k=k, filters=filters)
        return [
            SemanticRecord(
                id=str(h["id"]),
                text=str(h.get("payload", {}).get("text", "")),
                embedding=None,
                metadata=h.get("payload", {}),
            )
            for h in hits
        ]

    def _recall_world(
        self,
        query: str,
        k: int,
        time_range: tuple[datetime, datetime] | None,
    ) -> list[WorldNode]:
        assert self.world is not None
        valid_at = time_range[1] if time_range else None
        hits = self.world.search(query=query, k=k, valid_at=valid_at)
        return [
            WorldNode(
                id=str(h.get("id", "")),
                labels=list(h.get("labels", [])),
                properties=dict(h.get("properties", {})),
            )
            for h in hits
        ]
