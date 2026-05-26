"""Goal subsystem schemas.

См. `agent_architecture.html` § 4.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------- Enums ----------


class GoalClass(str, Enum):
    TASK = "task"
    INQUIRY = "inquiry"
    MAINTENANCE = "maintenance"
    STANDING = "standing"
    PRACTICE = "practice"


class GoalStatus(str, Enum):
    PENDING = "pending"
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DONE = "done"
    FAILED = "failed"
    ABANDONED = "abandoned"


class Origin(str, Enum):
    OPERATOR = "operator"
    DECOMPOSITION = "decomposition"
    SELF = "self"


class OriginSubtype(str, Enum):
    """Подтип для origin=self."""

    CURIOSITY = "curiosity"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"
    ANTICIPATORY = "anticipatory"


class Aggregation(str, Enum):
    """Для composite-предикатов (внутренние ноды дерева). v1: ALL и CUSTOM."""

    ALL = "all"
    CUSTOM = "custom"


# ---------- Predicate of success (discriminated union by `type`) ----------


class StructuralPredicate(BaseModel):
    """Точное соответствие выхода схеме/значению."""

    type: Literal["structural"] = "structural"
    expected_schema: dict[str, Any] = Field(
        description="JSON Schema, которой должен удовлетворять final_state"
    )


class StateChangePredicate(BaseModel):
    """Свойство среды стало истинно после выполнения."""

    type: Literal["state_change"] = "state_change"
    check_tool_id: str = Field(description="ID тула, который проверит состояние среды")
    expected_outcome: dict[str, Any] = Field(
        description="Ожидаемый возврат check_tool для успеха"
    )


class JudgePredicate(BaseModel):
    """LLM или человек судит по описанию."""

    type: Literal["judge"] = "judge"
    criterion: str = Field(description="Естественноязыковой критерий для судьи")


class CompositePredicate(BaseModel):
    """Агрегация над детьми (для внутренних нод дерева)."""

    type: Literal["composite"] = "composite"
    aggregation: Aggregation = Aggregation.ALL
    custom_check: str | None = Field(
        default=None, description="Описание custom-логики, если aggregation=custom"
    )


class ExternalPredicate(BaseModel):
    """Другой агент / система / время подтверждают (всегда deferred)."""

    type: Literal["external"] = "external"
    expected_signal: str = Field(
        description="Что должно появиться в sense, чтобы цель считалась завершённой"
    )


PredicateOfSuccess = Annotated[
    Union[
        StructuralPredicate,
        StateChangePredicate,
        JudgePredicate,
        CompositePredicate,
        ExternalPredicate,
    ],
    Field(discriminator="type"),
]


# ---------- Budget ----------


class Budget(BaseModel):
    """Бюджет на цель. None в любом поле — без лимита."""

    tokens: int | None = None
    steps: int | None = None
    time_seconds: float | None = None
    money: float | None = None

    tokens_consumed: int = 0
    steps_consumed: int = 0
    time_consumed: float = 0.0
    money_consumed: float = 0.0

    def is_exceeded(self) -> bool:
        if self.tokens is not None and self.tokens_consumed >= self.tokens:
            return True
        if self.steps is not None and self.steps_consumed >= self.steps:
            return True
        if self.time_seconds is not None and self.time_consumed >= self.time_seconds:
            return True
        if self.money is not None and self.money_consumed >= self.money:
            return True
        return False


# ---------- Goal ----------


class Goal(BaseModel):
    """Goal как первоклассный объект. См. § 4."""

    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(default_factory=uuid4)
    parent_id: UUID | None = None
    depends_on: list[UUID] = Field(default_factory=list)

    description: str
    # `class` — зарезервированное слово в Python; используем алиас.
    goal_class: GoalClass = Field(alias="class")
    predicate_of_success: PredicateOfSuccess

    priority: int = 0
    deadline: datetime | None = None

    status: GoalStatus = GoalStatus.PENDING
    budget: Budget = Field(default_factory=Budget)

    allowed_skills: list[str] = Field(default_factory=list)
    aggregation: Aggregation | None = None

    origin: Origin
    originator: str
    origin_subtype: OriginSubtype | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
