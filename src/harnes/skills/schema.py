"""Skill registry schemas.

См. `agent_architecture.html` § 9.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from harnes.goals.schema import GoalClass


class SkillStatus(str, Enum):
    ACTIVE = "active"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


class SkillOrigin(str, Enum):
    OPERATOR = "operator"
    REFLECT = "reflect"
    IMPORTED = "imported"


class FewShotExample(BaseModel):
    user: str
    assistant: str
    notes: str = ""


class SkillMetrics(BaseModel):
    invocation_count: int = 0
    success_rate: float = 0.0
    avg_cost_tokens: float = 0.0
    avg_steps: float = 0.0
    failure_modes: dict[str, int] = Field(default_factory=dict)
    warning_rate: float = 0.0


class Skill(BaseModel):
    """Скилл = prompt + allowed_tools + параметры. См. § 9."""

    id: str
    name: str
    description: str
    version: str = "0.0.1"
    parent_version_id: str | None = None

    prompt_template: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    required_inputs: dict[str, Any] = Field(default_factory=dict)

    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)

    # tool_id -> bool ИЛИ имя callable для conditional override.
    irreversibility_overrides: dict[str, Any] = Field(default_factory=dict)

    metrics: SkillMetrics = Field(default_factory=SkillMetrics)

    status: SkillStatus = SkillStatus.ACTIVE
    origin: SkillOrigin = SkillOrigin.OPERATOR
    applicable_goal_classes: list[GoalClass] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
