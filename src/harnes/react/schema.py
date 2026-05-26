"""ReAct schemas — Step types and Trajectory.

См. `agent_architecture.html` § 7, § 8.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------- Enums ----------


class ThoughtIntent(str, Enum):
    """Опциональный tag для Thought.text — не обязателен."""

    ANALYZE = "analyze"
    PLAN = "plan"
    DECIDE = "decide"
    REFLECT = "reflect"


class ObservationOutcome(str, Enum):
    """Outcome-таксономия для tool_layer. См. § 10."""

    SUCCESS = "success"
    TOOL_ERROR = "tool_error"
    SCHEMA_ERROR = "schema_error"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    MALFORMED_OUTPUT = "malformed_output"
    SEMANTIC_ERROR = "semantic_error"


class CritiqueVerdict(str, Enum):
    OK = "ok"
    WARNING = "warning"
    REJECT = "reject"


class TrajectoryStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    BUDGET_EXCEEDED = "budget_exceeded"
    ABANDONED = "abandoned"


# ---------- Cost ----------


class Cost(BaseModel):
    tokens: int = 0
    latency_seconds: float = 0.0


# ---------- Step types (discriminated union by `type`) ----------


class _StepBase(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cost: Cost = Field(default_factory=Cost)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThoughtStep(_StepBase):
    type: Literal["thought"] = "thought"
    text: str
    intent: ThoughtIntent | None = None


class PlanStepItem(BaseModel):
    description: str
    expected_tool: str | None = None


class PlanStep(_StepBase):
    type: Literal["plan"] = "plan"
    steps_intended: list[PlanStepItem]
    rationale: str
    revision_of: UUID | None = None


class ActionStep(_StepBase):
    type: Literal["action"] = "action"
    tool_id: str
    args: dict[str, Any]
    is_irreversible: bool = False
    expected_outcome: str | None = None


class ObservationStep(_StepBase):
    type: Literal["observation"] = "observation"
    outcome: ObservationOutcome
    payload: dict[str, Any] | None = None
    error_detail: str | None = None


class CritiqueStep(_StepBase):
    type: Literal["critique"] = "critique"
    target_step_id: UUID
    verdict: CritiqueVerdict
    reasoning: str
    recommendation: str | None = None
    risks_identified: list[str] = Field(default_factory=list)


class RetryNoteStep(_StepBase):
    type: Literal["retry_note"] = "retry_note"
    previous_trajectory_id: UUID
    failure_summary: str
    recommendation_for_this_attempt: str


Step = Annotated[
    Union[
        ThoughtStep,
        PlanStep,
        ActionStep,
        ObservationStep,
        CritiqueStep,
        RetryNoteStep,
    ],
    Field(discriminator="type"),
]


# ---------- Trajectory ----------


class Trajectory(BaseModel):
    """См. § 8. final_state полиморфен под predicate цели."""

    id: UUID = Field(default_factory=uuid4)
    goal_id: UUID
    parent_trajectory_id: UUID | None = None
    steps: list[Step] = Field(default_factory=list)
    status: TrajectoryStatus | None = None
    final_state: dict[str, Any] | str | None = None
    total_cost: Cost = Field(default_factory=Cost)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
