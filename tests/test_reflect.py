"""Tests for harnes.metacycle.reflect."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from harnes.goals.schema import (
    Goal,
    GoalClass,
    JudgePredicate,
    Origin,
)
from harnes.metacycle.reflect import (
    _bump_patch_version,
    _parse_reflect_json,
    reflect_failure_analysis,
)
from harnes.metacycle.schema import Verdict, VerifyStatus
from harnes.react.schema import (
    ActionStep,
    ObservationOutcome,
    ObservationStep,
    ThoughtStep,
    Trajectory,
    TrajectoryStatus,
)
from harnes.skills.schema import Skill
from harnes.skills.store import SkillRegistry


# ---------- helpers ----------


def test_bump_patch_version_semver() -> None:
    assert _bump_patch_version("0.0.1") == "0.0.2"
    assert _bump_patch_version("1.2.3") == "1.2.4"


def test_bump_patch_version_non_semver() -> None:
    assert _bump_patch_version("foo").startswith("foo+r")


def test_parse_reflect_json_direct() -> None:
    raw = '{"should_update": true, "diagnosis": "x", "new_prompt_template": "y"}'
    parsed = _parse_reflect_json(raw)
    assert parsed is not None
    assert parsed["should_update"] is True


def test_parse_reflect_json_garbage() -> None:
    assert _parse_reflect_json("no json here") is None


# ---------- helper fixtures ----------


@pytest.fixture
def goal() -> Goal:
    return Goal(
        description="do thing",
        goal_class=GoalClass.TASK,
        predicate_of_success=JudgePredicate(criterion="thing is done"),
        origin=Origin.OPERATOR,
        originator="test",
    )


@pytest.fixture
def failing_trajectory(goal: Goal) -> Trajectory:
    return Trajectory(
        goal_id=goal.id,
        steps=[
            ThoughtStep(text="trying to do thing"),
            ActionStep(tool_id="write_file", args={"path": "/tmp/x", "content": "?"}),
            ObservationStep(
                outcome=ObservationOutcome.TOOL_ERROR,
                error_detail="permission denied",
            ),
        ],
        status=TrajectoryStatus.FAILURE,
        final_state=None,
    )


@pytest.fixture
def fail_verdict() -> Verdict:
    return Verdict(
        status=VerifyStatus.FAIL,
        reasons=["thing not done"],
        measured_by="judge_llm",
    )


@pytest.fixture
def skill_registry(tmp_path: Path) -> SkillRegistry:
    reg = SkillRegistry(bundles_dir=tmp_path, metrics_db=":memory:")
    reg.save(
        Skill(
            id="general",
            name="general",
            description="g",
            version="0.0.1",
            prompt_template="Goal: {goal_description}\nTools:\n{tools_list}\nDo the thing.",
            allowed_tools=["read_file", "write_file"],
        )
    )
    return reg


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=50, completion_tokens=100)
    return response


# ---------- core logic ----------


def test_reflect_updates_skill_on_should_update_true(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
    skill_registry: SkillRegistry,
) -> None:
    new_template = (
        "Goal: {goal_description}\n"
        "Tools:\n{tools_list}\n"
        "IMPORTANT: check file permissions before writing. Do the thing."
    )
    payload = {
        "should_update": True,
        "diagnosis": "permission was denied",
        "new_prompt_template": new_template,
    }
    fake_llm = MagicMock(return_value=_mock_response(json.dumps(payload)))

    skill = skill_registry.get("general")
    assert skill is not None
    new_skill = reflect_failure_analysis(
        trajectory=failing_trajectory,
        goal=goal,
        verdict=fail_verdict,
        skill=skill,
        skill_registry=skill_registry,
        llm_call=fake_llm,
    )

    assert new_skill is not None
    assert new_skill.version == "0.0.2"
    assert new_skill.parent_version_id == "0.0.1"
    assert "permissions" in new_skill.prompt_template.lower()
    # Persisted
    reloaded = skill_registry.get("general")
    assert reloaded is not None
    assert reloaded.version == "0.0.2"

    # LLM called with tier=main
    assert fake_llm.call_args.kwargs["tier"] == "main"


def test_reflect_no_update_when_should_update_false(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
    skill_registry: SkillRegistry,
) -> None:
    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"should_update": false, "diagnosis": "environmental", '
            '"new_prompt_template": ""}'
        )
    )

    skill = skill_registry.get("general")
    assert skill is not None
    result = reflect_failure_analysis(
        trajectory=failing_trajectory,
        goal=goal,
        verdict=fail_verdict,
        skill=skill,
        skill_registry=skill_registry,
        llm_call=fake_llm,
    )
    assert result is None
    # Skill unchanged
    reloaded = skill_registry.get("general")
    assert reloaded is not None
    assert reloaded.version == "0.0.1"


def test_reflect_rejects_template_missing_placeholder(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
    skill_registry: SkillRegistry,
) -> None:
    """Если новый template потерял {goal_description} — отказываем."""
    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"should_update": true, "diagnosis": "x", '
            '"new_prompt_template": "Just do it. No placeholders."}'
        )
    )

    skill = skill_registry.get("general")
    assert skill is not None
    result = reflect_failure_analysis(
        trajectory=failing_trajectory,
        goal=goal,
        verdict=fail_verdict,
        skill=skill,
        skill_registry=skill_registry,
        llm_call=fake_llm,
    )
    assert result is None
    reloaded = skill_registry.get("general")
    assert reloaded.version == "0.0.1"


def test_reflect_handles_llm_exception(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
    skill_registry: SkillRegistry,
) -> None:
    fake_llm = MagicMock(side_effect=RuntimeError("net down"))

    skill = skill_registry.get("general")
    assert skill is not None
    result = reflect_failure_analysis(
        trajectory=failing_trajectory,
        goal=goal,
        verdict=fail_verdict,
        skill=skill,
        skill_registry=skill_registry,
        llm_call=fake_llm,
    )
    assert result is None  # fall-open


def test_reflect_handles_unparseable_response(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
    skill_registry: SkillRegistry,
) -> None:
    fake_llm = MagicMock(return_value=_mock_response("not json garbage"))

    skill = skill_registry.get("general")
    assert skill is not None
    result = reflect_failure_analysis(
        trajectory=failing_trajectory,
        goal=goal,
        verdict=fail_verdict,
        skill=skill,
        skill_registry=skill_registry,
        llm_call=fake_llm,
    )
    assert result is None


def test_reflect_skips_when_template_unchanged(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
    skill_registry: SkillRegistry,
) -> None:
    """should_update=true но new_template совпадает со старым → no-op."""
    skill = skill_registry.get("general")
    assert skill is not None
    payload = {
        "should_update": True,
        "diagnosis": "x",
        "new_prompt_template": skill.prompt_template,
    }
    fake_llm = MagicMock(return_value=_mock_response(json.dumps(payload)))
    result = reflect_failure_analysis(
        trajectory=failing_trajectory,
        goal=goal,
        verdict=fail_verdict,
        skill=skill,
        skill_registry=skill_registry,
        llm_call=fake_llm,
    )
    assert result is None
