"""Tests for harnes.react.critic + integration in ReAct loop."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from harnes.goals.schema import Goal, GoalClass, JudgePredicate, Origin
from harnes.react.critic import (
    _parse_critique_json,
    critique_action,
)
from harnes.react.loop import run_react
from harnes.react.schema import (
    ActionStep,
    CritiqueVerdict,
    ThoughtStep,
    Trajectory,
    TrajectoryStatus,
)
from harnes.skills.schema import Skill
from harnes.tools.registry import ToolRegistry, get_registry, reset_registry


def _make_goal() -> Goal:
    return Goal(
        description="test",
        goal_class=GoalClass.TASK,
        predicate_of_success=JudgePredicate(criterion="ok"),
        origin=Origin.OPERATOR,
        originator="test",
    )


def _mock_response(content: str, tokens: int = 20) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=tokens)
    return response


# ---------- _parse_critique_json ----------


def test_parse_critique_json_direct() -> None:
    raw = '{"verdict": "ok", "reasoning": "fine"}'
    assert _parse_critique_json(raw) == {"verdict": "ok", "reasoning": "fine"}


def test_parse_critique_json_with_prefix() -> None:
    raw = 'Critique: {"verdict": "reject", "reasoning": "no"}'
    assert _parse_critique_json(raw) == {"verdict": "reject", "reasoning": "no"}


def test_parse_critique_json_garbage() -> None:
    assert _parse_critique_json("just words") is None


# ---------- critique_action ----------


def test_critique_ok_verdict() -> None:
    goal = _make_goal()
    traj = Trajectory(goal_id=goal.id, steps=[ThoughtStep(text="planning")])
    action = ActionStep(tool_id="write_file", args={"path": "/tmp/x"}, is_irreversible=True)

    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"verdict": "ok", "reasoning": "fine", "risks": [], "recommendation": ""}'
        )
    )

    c = critique_action(action, traj, goal, llm_call=fake_llm)
    assert c.verdict == CritiqueVerdict.OK
    assert c.target_step_id == action.id
    assert "fine" in c.reasoning
    # Critic uses tier=light
    assert fake_llm.call_args.kwargs["tier"] == "light"


def test_critique_reject_verdict_with_recommendation() -> None:
    goal = _make_goal()
    traj = Trajectory(goal_id=goal.id)
    action = ActionStep(tool_id="write_file", args={"path": "/etc/passwd"}, is_irreversible=True)

    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"verdict": "reject", "reasoning": "writes to system file", '
            '"risks": ["data loss", "permission"], "recommendation": "use /tmp instead"}'
        )
    )

    c = critique_action(action, traj, goal, llm_call=fake_llm)
    assert c.verdict == CritiqueVerdict.REJECT
    assert "system file" in c.reasoning
    assert c.recommendation == "use /tmp instead"
    assert len(c.risks_identified) == 2


def test_critique_warning_verdict() -> None:
    goal = _make_goal()
    traj = Trajectory(goal_id=goal.id)
    action = ActionStep(tool_id="write_file", args={"path": "/tmp/y"}, is_irreversible=True)

    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"verdict": "warning", "reasoning": "might be ok"}'
        )
    )

    c = critique_action(action, traj, goal, llm_call=fake_llm)
    assert c.verdict == CritiqueVerdict.WARNING


def test_critique_unparseable_falls_open() -> None:
    """Если judge JSON невалидный — fall open (verdict=ok), не блокируем."""
    goal = _make_goal()
    traj = Trajectory(goal_id=goal.id)
    action = ActionStep(tool_id="write_file", args={"path": "/tmp/y"}, is_irreversible=True)

    fake_llm = MagicMock(return_value=_mock_response("not json"))

    c = critique_action(action, traj, goal, llm_call=fake_llm)
    assert c.verdict == CritiqueVerdict.OK
    assert "unparseable" in c.reasoning


def test_critique_llm_exception_falls_open() -> None:
    """Если LLM упал — fall open, action будет исполнен."""
    goal = _make_goal()
    traj = Trajectory(goal_id=goal.id)
    action = ActionStep(tool_id="write_file", args={"path": "/tmp/y"}, is_irreversible=True)

    fake_llm = MagicMock(side_effect=RuntimeError("net down"))

    c = critique_action(action, traj, goal, llm_call=fake_llm)
    assert c.verdict == CritiqueVerdict.OK
    assert "failed" in c.reasoning.lower()


def test_critique_unknown_verdict_normalises_to_ok() -> None:
    goal = _make_goal()
    traj = Trajectory(goal_id=goal.id)
    action = ActionStep(tool_id="write_file", args={"path": "/tmp/y"}, is_irreversible=True)

    fake_llm = MagicMock(
        return_value=_mock_response('{"verdict": "maybe", "reasoning": "?"}')
    )
    c = critique_action(action, traj, goal, llm_call=fake_llm)
    assert c.verdict == CritiqueVerdict.OK


# ---------- ReAct loop integration ----------


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
        prompt_template="Goal: {goal_description}\nTools:\n{tools_list}",
        allowed_tools=["read_file", "write_file"],
    )


def test_loop_critic_ok_lets_action_execute(
    tool_registry: ToolRegistry, general_skill: Skill, tmp_path
) -> None:
    """Critic ok → write_file исполняется (файл создаётся)."""
    target = tmp_path / "y.txt"
    target.write_text("old", encoding="utf-8")  # делает write_file irreversible

    goal = _make_goal()

    responses = iter(
        [
            _mock_response("plan to write"),
            _mock_response(
                f'{{"tool_id": "write_file", "args": {{"path": "{target}", "content": "new"}}}}'
            ),
            # Critic: ok
            _mock_response('{"verdict": "ok", "reasoning": "fine"}'),
            _mock_response("now finish"),
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
    assert target.read_text(encoding="utf-8") == "new"  # write выполнен
    # В trajectory есть critique-шаг
    assert any(s.type == "critique" for s in traj.steps)


def test_loop_critic_reject_skips_execution(
    tool_registry: ToolRegistry, general_skill: Skill, tmp_path
) -> None:
    """Critic reject → action НЕ исполняется, цикл возвращается к thought."""
    target = tmp_path / "y.txt"
    target.write_text("DO NOT OVERWRITE", encoding="utf-8")

    goal = _make_goal()

    responses = iter(
        [
            _mock_response("plan"),
            _mock_response(
                f'{{"tool_id": "write_file", "args": {{"path": "{target}", "content": "new"}}}}'
            ),
            # Critic: reject
            _mock_response(
                '{"verdict": "reject", "reasoning": "would destroy existing", '
                '"risks": ["data loss"], "recommendation": "use different path"}'
            ),
            # Back to thought → second action: finish
            _mock_response("ok, finishing without writing"),
            _mock_response(
                '{"tool_id": "finish", "args": {"final_state": {"aborted": true}}}'
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
    # Файл НЕ изменился
    assert target.read_text(encoding="utf-8") == "DO NOT OVERWRITE"
    # Critique-шаг с reject есть
    critiques = [s for s in traj.steps if s.type == "critique"]
    assert len(critiques) == 1
    assert critiques[0].verdict == CritiqueVerdict.REJECT


def test_loop_critic_too_many_rejects_fails_trajectory(
    tool_registry: ToolRegistry, general_skill: Skill, tmp_path
) -> None:
    """Превышение max_critic_rejections → trajectory FAILURE."""
    target = tmp_path / "y.txt"
    target.write_text("existing", encoding="utf-8")

    goal = _make_goal()

    # Каждый раз модель предлагает write_file → critic reject → loop повторяет
    # Очень много itераций — задействуем infinite generator
    def gen(messages, **kw):
        # Determine: is this a thought or action or critic call?
        # Last message in messages tells us context. Simpler: alternate.
        # Just return appropriate response based on counter.
        gen.counter += 1
        c = gen.counter
        # Pattern: thought, action(write), critique(reject), thought, action(write), critique(reject), ...
        mod = (c - 1) % 3
        if mod == 0:
            return _mock_response("must write")
        elif mod == 1:
            return _mock_response(
                f'{{"tool_id": "write_file", "args": {{"path": "{target}", "content": "x"}}}}'
            )
        else:
            return _mock_response('{"verdict": "reject", "reasoning": "no"}')

    gen.counter = 0
    mock_llm = MagicMock(side_effect=gen)

    traj = run_react(
        active_goal=goal,
        skill=general_skill,
        tool_registry=tool_registry,
        llm_call=mock_llm,
        max_critic_rejections=2,
        max_steps=20,
    )

    assert traj.status == TrajectoryStatus.FAILURE
    # Файл не изменился
    assert target.read_text(encoding="utf-8") == "existing"


def test_loop_critic_disabled_via_flag(
    tool_registry: ToolRegistry, general_skill: Skill, tmp_path
) -> None:
    """with_critic=False — critic не вызывается даже на irreversible."""
    target = tmp_path / "y.txt"
    target.write_text("old", encoding="utf-8")

    goal = _make_goal()

    responses = iter(
        [
            _mock_response("plan"),
            _mock_response(
                f'{{"tool_id": "write_file", "args": {{"path": "{target}", "content": "new"}}}}'
            ),
            # No critic call expected — straight to next thought
            _mock_response("done"),
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
        with_critic=False,
    )

    assert traj.status == TrajectoryStatus.SUCCESS
    assert target.read_text(encoding="utf-8") == "new"
    # Нет critique-шагов
    assert not any(s.type == "critique" for s in traj.steps)
