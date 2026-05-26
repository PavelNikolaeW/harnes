"""Tests for harnes.react.loop.

LLM-вызовы мокаются. Реальный e2e — в #13 smoke test.
"""
from __future__ import annotations

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
from harnes.react.loop import (
    FINISH_TOOL_ID,
    _detect_loop,
    _parse_action_json,
    run_react,
)
from harnes.react.schema import (
    ActionStep,
    ObservationOutcome,
    ThoughtStep,
    TrajectoryStatus,
)
from harnes.skills.schema import Skill
from harnes.tools.registry import ToolRegistry, get_registry, reset_registry


# ---------- Fixtures ----------


@pytest.fixture
def tool_registry() -> ToolRegistry:
    reset_registry()
    return get_registry()


@pytest.fixture
def general_skill() -> Skill:
    return Skill(
        id="general",
        name="general",
        description="general",
        prompt_template="Goal: {goal_description}\n\nTools:\n{tools_list}",
        allowed_tools=["read_file", "write_file"],
    )


def _make_goal(description: str = "test goal") -> Goal:
    return Goal(
        description=description,
        goal_class=GoalClass.TASK,
        predicate_of_success=JudgePredicate(criterion="ok"),
        origin=Origin.OPERATOR,
        originator="test",
    )


def _mock_response(content: str, tokens: int = 10) -> MagicMock:
    """Минимальный объект, похожий на ответ LiteLLM."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=tokens)
    return response


# ---------- Helpers ----------


def test_parse_action_json_direct() -> None:
    assert _parse_action_json('{"tool_id": "x", "args": {}}') == {
        "tool_id": "x",
        "args": {},
    }


def test_parse_action_json_with_prefix() -> None:
    assert _parse_action_json('Action: {"tool_id": "x", "args": {}}') == {
        "tool_id": "x",
        "args": {},
    }


def test_parse_action_json_garbage() -> None:
    assert _parse_action_json("just text no json") is None


# ---------- Loop detector ----------


def test_detect_loop_false_for_short() -> None:
    from harnes.react.schema import Trajectory

    traj = Trajectory(goal_id=_make_goal().id)
    traj.steps.append(ActionStep(tool_id="a", args={}))
    assert _detect_loop(traj, window=2) is False


def test_detect_loop_true_when_repeated() -> None:
    from harnes.react.schema import Trajectory

    traj = Trajectory(goal_id=_make_goal().id)
    # Один паттерн (a, b) повторяется дважды
    for _ in range(2):
        traj.steps.append(ActionStep(tool_id="a", args={"x": 1}))
        traj.steps.append(ActionStep(tool_id="b", args={"y": 2}))
    assert _detect_loop(traj, window=2) is True


# ---------- run_react: основные сценарии ----------


def test_react_finishes_on_finish_action(
    tool_registry: ToolRegistry, general_skill: Skill
) -> None:
    """Сразу же делает finish — траектория success, никаких тулов не вызвано."""
    goal = _make_goal("write hello.txt")

    responses = iter(
        [
            _mock_response("I should finish immediately."),
            _mock_response(
                '{"tool_id": "finish", "args": {"final_state": {"done": true}}}'
            ),
        ]
    )
    mock_llm = MagicMock(side_effect=lambda messages, **kw: next(responses))

    traj = run_react(
        active_goal=goal,
        skill=general_skill,
        tool_registry=tool_registry,
        llm_call=mock_llm,
    )

    assert traj.status == TrajectoryStatus.SUCCESS
    assert traj.final_state == {"done": True}
    # Шаги: 1 thought + 1 action = 2.
    assert len(traj.steps) == 2
    assert traj.steps[0].type == "thought"
    assert traj.steps[1].type == "action"


def test_react_uses_tool_then_finishes(
    tool_registry: ToolRegistry, general_skill: Skill, tmp_path: Path
) -> None:
    """Полный цикл: thought → read_file → observation → thought → finish."""
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    goal = _make_goal(f"read {src}")

    responses = iter(
        [
            _mock_response("Read the file first"),
            _mock_response(f'{{"tool_id": "read_file", "args": {{"path": "{src}"}}}}'),
            _mock_response("Got content. Now finishing."),
            _mock_response(
                '{"tool_id": "finish", "args": {"final_state": {"content": "hello"}}}'
            ),
        ]
    )
    mock_llm = MagicMock(side_effect=lambda messages, **kw: next(responses))

    traj = run_react(
        active_goal=goal,
        skill=general_skill,
        tool_registry=tool_registry,
        llm_call=mock_llm,
    )

    assert traj.status == TrajectoryStatus.SUCCESS
    # 2 thoughts + 1 action(read) + 1 observation + 1 action(finish) = 5
    assert len(traj.steps) == 5
    types = [s.type for s in traj.steps]
    assert types == ["thought", "action", "observation", "thought", "action"]
    assert traj.steps[2].outcome == ObservationOutcome.SUCCESS


def test_react_malformed_action_keeps_going(
    tool_registry: ToolRegistry, general_skill: Skill
) -> None:
    """Парсинг action провалился — добавляется malformed observation, цикл продолжается."""
    goal = _make_goal()

    responses = iter(
        [
            _mock_response("thinking"),
            _mock_response("not json at all"),  # malformed
            _mock_response("retry"),
            _mock_response(
                '{"tool_id": "finish", "args": {"final_state": {"done": true}}}'
            ),
        ]
    )
    mock_llm = MagicMock(side_effect=lambda messages, **kw: next(responses))

    traj = run_react(
        active_goal=goal,
        skill=general_skill,
        tool_registry=tool_registry,
        llm_call=mock_llm,
    )

    assert traj.status == TrajectoryStatus.SUCCESS
    # thought + malformed_obs + thought + finish_action = 4
    assert len(traj.steps) == 4
    assert traj.steps[1].outcome == ObservationOutcome.MALFORMED_OUTPUT


def test_react_terminates_on_max_steps(
    tool_registry: ToolRegistry, general_skill: Skill, tmp_path: Path
) -> None:
    """Без finish — упирается в max_steps, status=FAILURE."""
    src = tmp_path / "x.txt"
    src.write_text("y", encoding="utf-8")
    goal = _make_goal()

    # Бесконечный поток одинаковых thought/action — но loop_detector словит.
    # Чтобы тест проверял именно max_steps — используем разные args.
    counter = [0]

    def gen(messages: list[dict[str, Any]], **kw: Any) -> MagicMock:
        counter[0] += 1
        is_thought = counter[0] % 2 == 1
        if is_thought:
            return _mock_response(f"thought {counter[0]}")
        else:
            # action: read_file with rotating path (не будет считаться loop)
            return _mock_response(
                f'{{"tool_id": "read_file", "args": {{"path": "{src}_{counter[0]}"}}}}'
            )

    mock_llm = MagicMock(side_effect=gen)

    traj = run_react(
        active_goal=goal,
        skill=general_skill,
        tool_registry=tool_registry,
        llm_call=mock_llm,
        max_steps=3,
    )

    assert traj.status == TrajectoryStatus.FAILURE


def test_react_detects_loop(
    tool_registry: ToolRegistry, general_skill: Skill, tmp_path: Path
) -> None:
    """Повторяющиеся (tool_id, args) → loop detected → FAILURE."""
    src = tmp_path / "x.txt"
    src.write_text("y", encoding="utf-8")
    goal = _make_goal()

    # Один и тот же action в цикле.
    counter = [0]

    def gen(messages: list[dict[str, Any]], **kw: Any) -> MagicMock:
        counter[0] += 1
        if counter[0] % 2 == 1:
            return _mock_response(f"same thought {counter[0]}")
        return _mock_response(
            f'{{"tool_id": "read_file", "args": {{"path": "{src}"}}}}'
        )

    mock_llm = MagicMock(side_effect=gen)

    traj = run_react(
        active_goal=goal,
        skill=general_skill,
        tool_registry=tool_registry,
        llm_call=mock_llm,
        max_steps=20,
    )

    assert traj.status == TrajectoryStatus.FAILURE


def test_react_budget_exceeded(
    tool_registry: ToolRegistry, general_skill: Skill
) -> None:
    """Превышение токенов → status=BUDGET_EXCEEDED."""
    goal = _make_goal()

    # Каждый thought ест 10_000 токенов, бюджет 5_000 — упрётся сразу
    responses = iter(
        [
            _mock_response("expensive thought", tokens=10_000),
            _mock_response("never reached"),
        ]
    )
    mock_llm = MagicMock(side_effect=lambda messages, **kw: next(responses))

    traj = run_react(
        active_goal=goal,
        skill=general_skill,
        tool_registry=tool_registry,
        llm_call=mock_llm,
        budget_tokens=5_000,
    )

    assert traj.status == TrajectoryStatus.BUDGET_EXCEEDED
