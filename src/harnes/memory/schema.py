"""Memory layer schemas — unified recall API output.

См. `agent_architecture.html` § 13.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORLD = "world"


class EpisodicRecord(BaseModel):
    """Запись из episodic-лога (LanceDB)."""

    trajectory_id: UUID
    step_id: UUID
    goal_id: UUID
    timestamp: datetime
    step_type: str
    content: dict[str, Any]
    embedding: list[float] | None = None


class SemanticRecord(BaseModel):
    """Запись из vector-store (Qdrant)."""

    id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorldNode(BaseModel):
    """Узел/факт из temporal KG (Graphiti / Neo4j)."""

    id: str
    labels: list[str]
    properties: dict[str, Any] = Field(default_factory=dict)
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class ProceduralHint(BaseModel):
    """Подсказка по подходящему скиллу из procedural-store."""

    skill_id: str
    score: float
    rationale: str = ""


class MemoryBundle(BaseModel):
    """Что возвращает recall API. См. § 13."""

    episodic: list[EpisodicRecord] = Field(default_factory=list)
    semantic: list[SemanticRecord] = Field(default_factory=list)
    world: list[WorldNode] = Field(default_factory=list)
    procedural_hints: list[ProceduralHint] = Field(default_factory=list)
