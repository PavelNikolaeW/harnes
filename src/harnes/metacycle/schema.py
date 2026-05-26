"""Meta-cycle schemas — sense / attend / verify outputs.

См. `agent_architecture.html` § 3.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------- sense → ObservationBundle ----------


class SenseObservation(BaseModel):
    """Один элемент в ObservationBundle (входит из sense)."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str  # "cli" | "fs_watcher" | "timer" | ...
    payload: dict[str, Any] = Field(default_factory=dict)


class ObservationBundle(BaseModel):
    items: list[SenseObservation] = Field(default_factory=list)


# ---------- attend → FocusFrame ----------


class SalientItem(BaseModel):
    observation_id: UUID
    relevance: float
    novelty: float
    urgency: float
    score: float  # объединённый score (взвешенная сумма)


class FocusFrame(BaseModel):
    salient_items: list[SalientItem] = Field(default_factory=list)
    novelty_score: float = 0.0
    urgency_score: float = 0.0


# ---------- verify → Verdict ----------


class VerifyStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAIL = "fail"
    UNDETERMINED = "undetermined"


class Verdict(BaseModel):
    """Результат verify (immediate-leaf или composite)."""

    status: VerifyStatus
    reasons: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    # "structural" | "state_change" | "judge_llm" | ... — для аналитики метрик.
    measured_by: str | None = None
