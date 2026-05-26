"""Pydantic schema tests.

Покрывают: roundtrip-сериализация, дискриминируемые юнионы (Predicate, Step),
дефолты, простая бизнес-логика (Budget.is_exceeded).
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from harnes.goals.schema import (
    Aggregation,
    Budget,
    CompositePredicate,
    ExternalPredicate,
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
    StateChangePredicate,
    StructuralPredicate,
)
from harnes.memory.schema import MemoryBundle, SemanticRecord
from harnes.metacycle.schema import (
    FocusFrame,
    ObservationBundle,
    SenseObservation,
    Verdict,
    VerifyStatus,
)
from harnes.react.schema import (
    ActionStep,
    Cost,
    CritiqueStep,
    CritiqueVerdict,
    ObservationOutcome,
    ObservationStep,
    PlanStep,
    PlanStepItem,
    RetryNoteStep,
    ThoughtStep,
    Trajectory,
    TrajectoryStatus,
)
from harnes.skills.schema import Skill, SkillStatus
from harnes.tools.schema import BaseIrreversibility, Tool, ToolCategory


# ---------- Goal / Predicate ----------


def test_goal_with_structural_predicate() -> None:
    goal = Goal(
        description="write a haiku",
        goal_class=GoalClass.TASK,
        predicate_of_success=StructuralPredicate(
            expected_schema={"type": "object", "properties": {"text": {"type": "string"}}}
        ),
        origin=Origin.OPERATOR,
        originator="pavel",
    )
    assert goal.goal_class == GoalClass.TASK
    assert goal.predicate_of_success.type == "structural"
    assert goal.status == GoalStatus.PENDING


def test_goal_class_alias_in_dict_input() -> None:
    """Можно создавать через alias `class`, а не только через goal_class."""
    goal = Goal.model_validate(
        {
            "description": "x",
            "class": "task",  # alias
            "predicate_of_success": {"type": "judge", "criterion": "ok if X"},
            "origin": "operator",
            "originator": "pavel",
        }
    )
    assert goal.goal_class == GoalClass.TASK


def test_predicate_dispatch_state_change() -> None:
    """Discriminated union должен подобрать правильный класс по `type`."""
    goal = Goal.model_validate(
        {
            "description": "ensure file exists",
            "class": "task",
            "predicate_of_success": {
                "type": "state_change",
                "check_tool_id": "file_exists",
                "expected_outcome": {"exists": True},
            },
            "origin": "operator",
            "originator": "pavel",
        }
    )
    assert isinstance(goal.predicate_of_success, StateChangePredicate)


def test_predicate_dispatch_composite() -> None:
    goal = Goal(
        description="meta",
        goal_class=GoalClass.TASK,
        predicate_of_success=CompositePredicate(aggregation=Aggregation.ALL),
        origin=Origin.OPERATOR,
        originator="pavel",
    )
    assert isinstance(goal.predicate_of_success, CompositePredicate)
    assert goal.predicate_of_success.aggregation == Aggregation.ALL


def test_predicate_dispatch_external() -> None:
    p = ExternalPredicate(expected_signal="webhook from CI")
    assert p.type == "external"


def test_judge_predicate() -> None:
    p = JudgePredicate(criterion="response is polite")
    assert p.criterion == "response is polite"


def test_goal_roundtrip() -> None:
    """Сериализация → JSON dict → десериализация должна сохранить тип предиката."""
    original = Goal(
        description="ping",
        goal_class=GoalClass.INQUIRY,
        predicate_of_success=JudgePredicate(criterion="agent answered something"),
        origin=Origin.OPERATOR,
        originator="pavel",
    )
    dumped = original.model_dump(mode="json", by_alias=True)
    restored = Goal.model_validate(dumped)
    assert restored.goal_class == GoalClass.INQUIRY
    assert isinstance(restored.predicate_of_success, JudgePredicate)


def test_budget_exceeded() -> None:
    b = Budget(tokens=100, tokens_consumed=50)
    assert not b.is_exceeded()
    b.tokens_consumed = 100
    assert b.is_exceeded()

    b2 = Budget(steps=10)
    b2.steps_consumed = 11
    assert b2.is_exceeded()


def test_budget_no_limits_never_exceeded() -> None:
    b = Budget()
    b.tokens_consumed = 1_000_000
    assert not b.is_exceeded()


# ---------- Step types ----------


def test_thought_step() -> None:
    s = ThoughtStep(text="I should read the file")
    assert s.type == "thought"
    assert s.cost.tokens == 0


def test_action_step() -> None:
    a = ActionStep(tool_id="read_file", args={"path": "/tmp/x"})
    assert a.type == "action"
    assert a.is_irreversible is False


def test_observation_step_with_error() -> None:
    o = ObservationStep(
        outcome=ObservationOutcome.TOOL_ERROR,
        error_detail="permission denied",
    )
    assert o.outcome == ObservationOutcome.TOOL_ERROR


def test_critique_step() -> None:
    c = CritiqueStep(
        target_step_id=uuid4(),
        verdict=CritiqueVerdict.REJECT,
        reasoning="this would destroy the file",
        risks_identified=["data loss"],
    )
    assert c.verdict == CritiqueVerdict.REJECT


def test_plan_step() -> None:
    p = PlanStep(
        steps_intended=[
            PlanStepItem(description="read X", expected_tool="read_file"),
            PlanStepItem(description="write Y", expected_tool="write_file"),
        ],
        rationale="task requires copy",
    )
    assert len(p.steps_intended) == 2


def test_retry_note_step() -> None:
    r = RetryNoteStep(
        previous_trajectory_id=uuid4(),
        failure_summary="schema validation failed",
        recommendation_for_this_attempt="use correct field names",
    )
    assert r.type == "retry_note"


# ---------- Trajectory + step union dispatch ----------


def test_trajectory_with_mixed_steps_roundtrip() -> None:
    goal_id = uuid4()
    traj = Trajectory(
        goal_id=goal_id,
        steps=[
            ThoughtStep(text="plan"),
            ActionStep(tool_id="t", args={"k": "v"}),
            ObservationStep(outcome=ObservationOutcome.SUCCESS, payload={"ok": True}),
        ],
    )

    dumped = traj.model_dump(mode="json")
    restored = Trajectory.model_validate(dumped)

    assert len(restored.steps) == 3
    assert restored.steps[0].type == "thought"
    assert restored.steps[1].type == "action"
    assert restored.steps[2].type == "observation"
    # Type preservation through union dispatch
    assert isinstance(restored.steps[1], ActionStep)


def test_trajectory_defaults() -> None:
    t = Trajectory(goal_id=uuid4())
    assert t.steps == []
    assert t.status is None
    assert t.ended_at is None


# ---------- Tool ----------


def test_tool_basic() -> None:
    t = Tool(
        id="read_file",
        name="read_file",
        description="read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        output_schema={"type": "object"},
        category=ToolCategory.INFO,
        implementation_ref="harnes.tools.builtin.read_file",
    )
    assert t.base_irreversibility == BaseIrreversibility.NEVER
    assert t.retry_policy.max_retries == 3


# ---------- Skill ----------


def test_skill_basic() -> None:
    s = Skill(
        id="general",
        name="general",
        description="default skill",
        prompt_template="You are a helpful agent. {goal_description}",
        allowed_tools=["read_file", "write_file"],
        applicable_goal_classes=[GoalClass.TASK],
    )
    assert s.status == SkillStatus.ACTIVE
    assert GoalClass.TASK in s.applicable_goal_classes


# ---------- Memory ----------


def test_memory_bundle_empty() -> None:
    b = MemoryBundle()
    assert b.episodic == []
    assert b.semantic == []


def test_memory_bundle_with_semantic_record() -> None:
    rec = SemanticRecord(id="x", text="hello", embedding=[0.1, 0.2])
    b = MemoryBundle(semantic=[rec])
    assert len(b.semantic) == 1


# ---------- Metacycle outputs ----------


def test_observation_bundle() -> None:
    bundle = ObservationBundle(
        items=[SenseObservation(source="cli", payload={"goal": "x"})]
    )
    assert bundle.items[0].source == "cli"


def test_focus_frame_defaults() -> None:
    f = FocusFrame()
    assert f.novelty_score == 0.0
    assert f.salient_items == []


def test_verdict() -> None:
    v = Verdict(
        status=VerifyStatus.SUCCESS,
        reasons=["schema match"],
        measured_by="structural",
    )
    assert v.status == VerifyStatus.SUCCESS


def test_cost() -> None:
    c = Cost(tokens=42, latency_seconds=1.5)
    assert c.tokens == 42
