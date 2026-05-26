"""Tests for harnes.eval.harness.

Используем in-test stub-adapter — реальный MemoryAgentBench не дёргаем.
"""
from __future__ import annotations

from typing import Iterable

import pytest

from harnes.eval import BenchmarkAdapter, EvalResult, PerTaskResult, run_evaluation
from harnes.goals.schema import (
    Goal,
    GoalClass,
    JudgePredicate,
    Origin,
)
from harnes.react.schema import Cost, Trajectory, TrajectoryStatus


# ---------- Stub adapter ----------


class _StubAdapter:
    """Минимальный adapter с двумя задачами и фиксированной верификацией."""

    name = "stub"

    def __init__(self, verify_results: dict[str, tuple[bool, str | None]]) -> None:
        self._verify_results = verify_results

    def tasks(self) -> Iterable[tuple[str, Goal]]:
        yield "t1", Goal(
            description="task 1",
            goal_class=GoalClass.TASK,
            predicate_of_success=JudgePredicate(criterion="ok"),
            origin=Origin.OPERATOR,
            originator="eval",
        )
        yield "t2", Goal(
            description="task 2",
            goal_class=GoalClass.TASK,
            predicate_of_success=JudgePredicate(criterion="ok"),
            origin=Origin.OPERATOR,
            originator="eval",
        )

    def verify(
        self, task_id: str, trajectory: Trajectory
    ) -> tuple[bool, str | None]:
        return self._verify_results.get(task_id, (False, "unknown"))


def _fake_agent(success: bool = True, tokens: int = 100, steps: int = 3):
    def runner(goal: Goal) -> Trajectory:
        traj = Trajectory(goal_id=goal.id, total_cost=Cost(tokens=tokens))
        traj.status = (
            TrajectoryStatus.SUCCESS if success else TrajectoryStatus.FAILURE
        )
        # Добавим dummy steps чтобы len(steps) == steps
        from harnes.react.schema import ThoughtStep

        for _ in range(steps):
            traj.steps.append(ThoughtStep(text="..."))
        return traj
    return runner


# ---------- Tests ----------


def test_run_evaluation_all_success() -> None:
    adapter = _StubAdapter(verify_results={"t1": (True, None), "t2": (True, None)})
    result = run_evaluation(adapter, _fake_agent(success=True, tokens=100, steps=3))

    assert result.name == "stub"
    assert len(result.per_task) == 2
    assert result.success_rate == 1.0
    assert result.avg_cost_tokens == 100
    assert result.avg_steps == 3
    assert result.failure_modes == {}


def test_run_evaluation_mixed() -> None:
    adapter = _StubAdapter(
        verify_results={"t1": (True, None), "t2": (False, "schema_error")}
    )
    result = run_evaluation(adapter, _fake_agent())
    assert result.success_rate == 0.5
    assert result.failure_modes == {"schema_error": 1}


def test_run_evaluation_agent_crash() -> None:
    adapter = _StubAdapter(verify_results={"t1": (True, None), "t2": (True, None)})

    def crashy(goal: Goal) -> Trajectory:
        raise RuntimeError("boom")

    result = run_evaluation(adapter, crashy)
    assert result.success_rate == 0.0
    assert all(r.failure_mode and r.failure_mode.startswith("agent_crash") for r in result.per_task)


def test_run_evaluation_limit() -> None:
    adapter = _StubAdapter(verify_results={"t1": (True, None), "t2": (True, None)})
    result = run_evaluation(adapter, _fake_agent(), limit=1)
    assert len(result.per_task) == 1
    assert result.per_task[0].task_id == "t1"


def test_eval_result_empty_metrics() -> None:
    result = EvalResult(name="empty")
    assert result.success_rate == 0.0
    assert result.avg_steps == 0.0
    assert result.avg_cost_tokens == 0.0
