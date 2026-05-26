"""Semantic vector store on Qdrant.

См. `agent_architecture.html` § 13.

В v0 — простая обёртка: одна коллекция `semantic_facts`, upsert + search.
Эмбеддинги ждём от harnes.llm.embeddings.embed(). Расширения (мульти-коллекции,
filter-policy на decay) — отдельная задача после v0.
"""
from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

log = structlog.get_logger()


DEFAULT_COLLECTION = "semantic_facts"
DEFAULT_DIM = 1024  # BGE-M3


class SemanticStore:
    """Qdrant-обёртка для семантической памяти."""

    def __init__(
        self,
        url: str,
        collection: str = DEFAULT_COLLECTION,
        dim: int = DEFAULT_DIM,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim

    def ensure_collection(self) -> None:
        """Создаёт коллекцию, если её нет. Идемпотентно."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            log.info(
                "semantic.collection.created",
                collection=self.collection,
                dim=self.dim,
            )

    def upsert(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        """records: [{"id": ..., "vector": [...], "payload": {...}}, ...]"""
        if not records:
            return
        points = [
            PointStruct(id=r["id"], vector=r["vector"], payload=r.get("payload", {}))
            for r in records
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Возвращает list of {id, score, payload}."""
        # v0: filters игнорируем (потом — Qdrant filter DSL).
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k,
        )
        return [
            {"id": h.id, "score": h.score, "payload": h.payload}
            for h in hits
        ]
