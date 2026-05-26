"""Tool layer schemas.

См. `agent_architecture.html` § 10.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    INFO = "info"
    ACTION = "action"
    COMMUNICATION = "communication"
    INTERNAL = "internal"
    META = "meta"


class BaseIrreversibility(str, Enum):
    NEVER = "never"
    CONDITIONAL = "conditional"
    ALWAYS = "always"


class BackoffStrategy(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class RetryPolicy(BaseModel):
    """Что ретраить и как. retryable_outcomes — список значений ObservationOutcome."""

    retryable_outcomes: list[str] = Field(default_factory=list)
    max_retries: int = 3
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    initial_delay_seconds: float = 1.0


class ToolMetrics(BaseModel):
    invocation_count: int = 0
    success_count: int = 0
    failure_count_by_outcome: dict[str, int] = Field(default_factory=dict)
    avg_latency_seconds: float = 0.0


class Tool(BaseModel):
    """Описание тула в глобальном реестре. См. § 10."""

    id: str
    name: str
    description: str

    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    base_irreversibility: BaseIrreversibility = BaseIrreversibility.NEVER
    # Имя callable из реестра предикатов, если base_irreversibility=CONDITIONAL.
    conditional_predicate: str | None = None

    side_effects: str = ""
    category: ToolCategory = ToolCategory.INFO

    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: float = 30.0
    # "module.fn" или другой routing-ключ для tool-layer dispatcher.
    implementation_ref: str

    metrics: ToolMetrics = Field(default_factory=ToolMetrics)
