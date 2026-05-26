"""Tests for harnes.metacycle.verifiers.

Judge-вызовы LLM мокаются. Реальный judge — в test_e2e_smoke.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from harnes.goals.schema import (
    CompositePredicate,
    ExternalPredicate,
    Goal,
    GoalClass,
    JudgePredicate,
    Origin,
    StateChangePredicate,
    StructuralPredicate,
)
from harnes.metacycle.schema import VerifyStatus
from harnes.metacycle.verifiers import (
    _parse_judge_json,
    verify,
    verify_composite,
    verify_external,
    verify_judge,
    verify_state_change,
    verify_structural,
)
from harnes.react.schema import (
    ActionStep,
    ObservationOutcome,
    ObservationStep,
    ThoughtStep,
    Trajectory,
    TrajectoryStatus,
)


def _goal_with(predicate) -> Goal:
    return Goal(
        description="test",
        goal_class=GoalClass.TASK,
        predicate_of_success=predicate,
        origin=Origin.OPERATOR,
        originator="test",
    )


def _traj_with_final(final_state) -> Trajectory:
    goal_id = _goal_with(JudgePredicate(criterion="x")).id
    return Trajectory(
        goal_id=goal_id,
        status=TrajectoryStatus.SUCCESS,
        final_state=final_state,
        steps=[ThoughtStep(text="t")],
    )


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    return response


# ---------- helpers ----------


def test_parse_judge_json_direct() -> None:
    assert _parse_judge_json('{"success": true, "reasoning": "ok"}') == {
        "success": True,
        "reasoning": "ok",
    }


def test_parse_judge_json_with_prefix() -> None:
    assert _parse_judge_json('Verdict: {"success": false, "reasoning": "no"}') == {
        "success": False,
        "reasoning": "no",
    }


def test_parse_judge_json_garbage() -> None:
    assert _parse_judge_json("just words") is None


# ---------- structural ----------


def test_structural_success_when_final_state() -> None:
    goal = _goal_with(StructuralPredicate(expected_schema={"type": "object"}))
    traj = _traj_with_final({"ok": True})
    v = verify_structural(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.SUCCESS
    assert v.measured_by == "structural"


def test_structural_fail_when_no_final_state() -> None:
    goal = _goal_with(StructuralPredicate(expected_schema={"type": "object"}))
    traj = _traj_with_final(None)
    v = verify_structural(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.FAIL


# ---------- judge ----------


def test_judge_success_path() -> None:
    goal = _goal_with(JudgePredicate(criterion="file contains hello"))
    traj = _traj_with_final({"content": "hello"})

    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"success": true, "reasoning": "file content matches criterion"}'
        )
    )

    v = verify_judge(goal.predicate_of_success, traj, goal, llm_call=fake_llm)

    assert v.status == VerifyStatus.SUCCESS
    assert v.measured_by == "judge_llm"
    assert "matches criterion" in (v.reasons[0] if v.reasons else "")
    # Judge LLM был вызван с tier=light (анти-коррелирован с executor)
    call_kwargs = fake_llm.call_args.kwargs
    assert call_kwargs["tier"] == "light"


def test_judge_fail_path() -> None:
    goal = _goal_with(JudgePredicate(criterion="file contains hello"))
    traj = _traj_with_final({"content": "wrong"})

    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"success": false, "reasoning": "wrong content"}'
        )
    )

    v = verify_judge(goal.predicate_of_success, traj, goal, llm_call=fake_llm)
    assert v.status == VerifyStatus.FAIL
    assert v.measured_by == "judge_llm"


def test_judge_malformed_response_undetermined() -> None:
    goal = _goal_with(JudgePredicate(criterion="x"))
    traj = _traj_with_final({"ok": True})

    fake_llm = MagicMock(return_value=_mock_response("not json"))

    v = verify_judge(goal.predicate_of_success, traj, goal, llm_call=fake_llm)
    assert v.status == VerifyStatus.UNDETERMINED
    assert v.measured_by == "judge_llm"


def test_judge_llm_exception_undetermined() -> None:
    goal = _goal_with(JudgePredicate(criterion="x"))
    traj = _traj_with_final({"ok": True})

    fake_llm = MagicMock(side_effect=RuntimeError("connection refused"))

    v = verify_judge(goal.predicate_of_success, traj, goal, llm_call=fake_llm)
    assert v.status == VerifyStatus.UNDETERMINED
    assert "connection refused" in (v.reasons[0] if v.reasons else "")


def test_judge_sees_recent_steps_in_prompt() -> None:
    goal = _goal_with(JudgePredicate(criterion="x"))
    traj = Trajectory(
        goal_id=goal.id,
        final_state={"x": 1},
        steps=[
            ThoughtStep(text="planning"),
            ActionStep(tool_id="write_file", args={"path": "/tmp/y"}),
            ObservationStep(outcome=ObservationOutcome.SUCCESS, payload={"bytes_written": 5}),
        ],
    )

    captured: dict[str, Any] = {}

    def fake_llm(messages, **kw):
        captured["messages"] = messages
        return _mock_response('{"success": true, "reasoning": "ok"}')

    verify_judge(goal.predicate_of_success, traj, goal, llm_call=fake_llm)

    # User-промпт должен содержать запись о действии и наблюдении.
    user_content = captured["messages"][1]["content"]
    assert "write_file" in user_content
    assert "bytes_written" in user_content


# ---------- state_change / composite — stubs ----------


def test_state_change_unknown_tool_is_undetermined() -> None:
    goal = _goal_with(
        StateChangePredicate(
            check_tool_id="nonexistent_tool", expected_outcome={"x": 1}
        )
    )
    traj = _traj_with_final(None)
    v = verify_state_change(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.UNDETERMINED
    assert v.measured_by == "state_change"
    assert "not in registry" in v.reasons[0]


def test_state_change_success_via_read_file(tmp_path) -> None:
    """Реальная проверка через builtin read_file: создаём файл с известным
    содержимым, верификатор проверяет."""
    from harnes.tools.registry import get_registry, reset_registry

    reset_registry()
    get_registry()  # auto-registers builtin tools

    target = tmp_path / "state.txt"
    target.write_text("hello world", encoding="utf-8")

    goal = _goal_with(
        StateChangePredicate(
            check_tool_id="read_file",
            check_tool_args={"path": str(target)},
            expected_outcome={"content": "hello world"},
        )
    )
    traj = _traj_with_final({"done": True})
    v = verify_state_change(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.SUCCESS, v.reasons
    assert v.measured_by == "state_change"


def test_state_change_fail_when_content_mismatch(tmp_path) -> None:
    from harnes.tools.registry import get_registry, reset_registry

    reset_registry()
    get_registry()

    target = tmp_path / "state.txt"
    target.write_text("WRONG content", encoding="utf-8")

    goal = _goal_with(
        StateChangePredicate(
            check_tool_id="read_file",
            check_tool_args={"path": str(target)},
            expected_outcome={"content": "expected text"},
        )
    )
    traj = _traj_with_final({"done": True})
    v = verify_state_change(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.FAIL
    assert "mismatch" in v.reasons[0]


def test_state_change_fail_when_tool_errors(tmp_path) -> None:
    """Если check_tool сам упал (например, файл не существует) — FAIL."""
    from harnes.tools.registry import get_registry, reset_registry

    reset_registry()
    get_registry()

    goal = _goal_with(
        StateChangePredicate(
            check_tool_id="read_file",
            check_tool_args={"path": str(tmp_path / "missing.txt")},
            expected_outcome={"content": "anything"},
        )
    )
    traj = _traj_with_final(None)
    v = verify_state_change(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.FAIL
    assert v.measured_by == "state_change"


def test_state_change_subset_match_works(tmp_path) -> None:
    """expected_outcome — подмножество полей, лишние ключи в payload OK."""
    from harnes.tools.registry import get_registry, reset_registry

    reset_registry()
    get_registry()

    target = tmp_path / "x.txt"
    target.write_text("abc", encoding="utf-8")

    # read_file возвращает {content, bytes_read, truncated} — мы проверяем только content
    goal = _goal_with(
        StateChangePredicate(
            check_tool_id="read_file",
            check_tool_args={"path": str(target)},
            expected_outcome={"content": "abc"},  # bytes_read/truncated не указаны
        )
    )
    traj = _traj_with_final({"done": True})
    v = verify_state_change(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.SUCCESS


def test_composite_without_repo_is_undetermined() -> None:
    """В v0.3 composite реализован, но без goal_repo он не может опросить children."""
    goal = _goal_with(CompositePredicate())
    traj = _traj_with_final(None)
    v = verify_composite(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.UNDETERMINED
    assert "needs goal_repo" in v.reasons[0]


def test_external_is_undetermined() -> None:
    goal = _goal_with(ExternalPredicate(expected_signal="webhook"))
    traj = _traj_with_final(None)
    v = verify_external(goal.predicate_of_success, traj, goal)
    assert v.status == VerifyStatus.UNDETERMINED
    assert v.measured_by == "external"


# ---------- dispatcher ----------


def test_verify_dispatcher_routes_to_structural() -> None:
    goal = _goal_with(StructuralPredicate(expected_schema={}))
    traj = _traj_with_final({"ok": True})
    v = verify(traj, goal)
    assert v.measured_by == "structural"


def test_verify_dispatcher_routes_to_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    goal = _goal_with(JudgePredicate(criterion="x"))
    traj = _traj_with_final({"ok": True})

    fake_llm = MagicMock(
        return_value=_mock_response('{"success": true, "reasoning": "ok"}')
    )

    v = verify(traj, goal, llm_call=fake_llm)
    assert v.measured_by == "judge_llm"
    assert v.status == VerifyStatus.SUCCESS


def test_verify_dispatcher_routes_to_external() -> None:
    goal = _goal_with(ExternalPredicate(expected_signal="x"))
    traj = _traj_with_final({"ok": True})
    v = verify(traj, goal)
    assert v.measured_by == "external"
