"""Tests for harnes.skills.store."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harnes.goals.schema import GoalClass
from harnes.skills.schema import Skill, SkillStatus
from harnes.skills.store import SkillRegistry


def _write_skill(dir_path: Path, skill: Skill) -> None:
    data = skill.model_dump(mode="json", exclude_none=True)
    (dir_path / f"{skill.id}.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


@pytest.fixture
def registry(tmp_path: Path) -> SkillRegistry:
    return SkillRegistry(bundles_dir=tmp_path, metrics_db=":memory:")


def test_load_general_skill_from_repo() -> None:
    """Проверяем, что дефолтный general.yaml из репо валиден и загружается."""
    repo_skills = Path(__file__).resolve().parent.parent / "skills"
    reg = SkillRegistry(bundles_dir=repo_skills, metrics_db=":memory:")
    general = reg.get("general")
    assert general is not None
    assert general.name == "general"
    assert "read_file" in general.allowed_tools
    assert "write_file" in general.allowed_tools
    assert GoalClass.TASK in general.applicable_goal_classes


def test_load_all_empty_dir_returns_empty(registry: SkillRegistry) -> None:
    assert registry.load_all() == []


def test_save_and_get(registry: SkillRegistry) -> None:
    skill = Skill(
        id="foo",
        name="foo",
        description="test skill",
        prompt_template="do {goal_description}",
        allowed_tools=["t1"],
    )
    registry.save(skill)

    fetched = registry.get("foo")
    assert fetched is not None
    assert fetched.id == "foo"
    assert fetched.prompt_template == "do {goal_description}"


def test_list_active_filters_deprecated(registry: SkillRegistry) -> None:
    registry.save(
        Skill(
            id="a",
            name="a",
            description="x",
            prompt_template="...",
            status=SkillStatus.ACTIVE,
        )
    )
    registry.save(
        Skill(
            id="b",
            name="b",
            description="x",
            prompt_template="...",
            status=SkillStatus.DEPRECATED,
        )
    )
    active = registry.list_active()
    assert {s.id for s in active} == {"a"}


def test_list_applicable_by_goal_class(registry: SkillRegistry) -> None:
    registry.save(
        Skill(
            id="general",
            name="general",
            description="any",
            prompt_template="...",
            applicable_goal_classes=[],  # пусто = применим ко всем
        )
    )
    registry.save(
        Skill(
            id="task_only",
            name="task_only",
            description="only task",
            prompt_template="...",
            applicable_goal_classes=[GoalClass.TASK],
        )
    )
    registry.save(
        Skill(
            id="inquiry_only",
            name="inquiry_only",
            description="only inquiry",
            prompt_template="...",
            applicable_goal_classes=[GoalClass.INQUIRY],
        )
    )

    for_task = registry.list_applicable(GoalClass.TASK)
    for_inquiry = registry.list_applicable(GoalClass.INQUIRY)

    assert {s.id for s in for_task} == {"general", "task_only"}
    assert {s.id for s in for_inquiry} == {"general", "inquiry_only"}


def test_get_missing_returns_none(registry: SkillRegistry) -> None:
    assert registry.get("nonexistent") is None


# ---------- Metrics ----------


def test_metrics_empty_returns_zero(registry: SkillRegistry) -> None:
    metrics = registry.get_metrics("general")
    assert metrics.invocation_count == 0
    assert metrics.success_rate == 0.0


def test_record_invocation_and_aggregate(registry: SkillRegistry) -> None:
    registry.record_invocation(
        "general", "0.0.1", success=True, cost_tokens=100, steps=3
    )
    registry.record_invocation(
        "general", "0.0.1", success=True, cost_tokens=200, steps=5
    )
    registry.record_invocation(
        "general",
        "0.0.1",
        success=False,
        cost_tokens=50,
        steps=2,
        failure_mode="tool_error",
    )

    m = registry.get_metrics("general")
    assert m.invocation_count == 3
    assert m.success_rate == pytest.approx(2 / 3)
    assert m.avg_cost_tokens == pytest.approx((100 + 200 + 50) / 3)
    assert m.avg_steps == pytest.approx((3 + 5 + 2) / 3)
    assert m.failure_modes == {"tool_error": 1}


def test_metrics_per_version(registry: SkillRegistry) -> None:
    registry.record_invocation("s", "0.0.1", success=True, cost_tokens=100)
    registry.record_invocation("s", "0.0.2", success=False, cost_tokens=200)

    v1 = registry.get_metrics("s", version="0.0.1")
    v2 = registry.get_metrics("s", version="0.0.2")
    assert v1.invocation_count == 1
    assert v1.success_rate == 1.0
    assert v2.invocation_count == 1
    assert v2.success_rate == 0.0


def test_warning_rate(registry: SkillRegistry) -> None:
    registry.record_invocation("s", "0.0.1", success=True, warning=False)
    registry.record_invocation("s", "0.0.1", success=True, warning=True)
    registry.record_invocation("s", "0.0.1", success=True, warning=True)

    m = registry.get_metrics("s")
    assert m.warning_rate == pytest.approx(2 / 3)
