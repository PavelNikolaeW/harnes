"""Tests for harnes.eval.adapters.memory_agent_bench."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from harnes.eval import MemoryAgentBenchAdapter, run_evaluation
from harnes.goals.schema import GoalClass
from harnes.react.schema import Cost, Trajectory, TrajectoryStatus


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "memory_agent_bench_sample.json"


@pytest.fixture
def adapter() -> MemoryAgentBenchAdapter:
    return MemoryAgentBenchAdapter(tasks_file=FIXTURE_PATH)


# ---------- Construction ----------


def test_adapter_loads_from_file(adapter: MemoryAgentBenchAdapter) -> None:
    assert adapter.name == "memory_agent_bench_sample"
    assert adapter.metric == "substring_exact_match"
    tasks = list(adapter.tasks())
    assert len(tasks) == 5


def test_adapter_loads_from_inline_tasks() -> None:
    tasks = [{"task_id": "x", "context": "y", "question": "?", "expected_answer": "z"}]
    a = MemoryAgentBenchAdapter(tasks=tasks)
    items = list(a.tasks())
    assert len(items) == 1


def test_adapter_requires_input() -> None:
    with pytest.raises(ValueError):
        MemoryAgentBenchAdapter()


# ---------- Goal generation ----------


def test_goals_are_task_class_with_judge_predicate(
    adapter: MemoryAgentBenchAdapter,
) -> None:
    for tid, goal in adapter.tasks():
        assert goal.goal_class == GoalClass.TASK
        assert goal.predicate_of_success.type == "judge"
        assert goal.metadata["task_id"] == tid
        assert goal.metadata["benchmark"] == "memory_agent_bench_sample"
        assert "context" not in goal.description.lower()[:20]  # context inline
        assert "Question:" in goal.description


def test_goal_description_contains_context_and_question(
    adapter: MemoryAgentBenchAdapter,
) -> None:
    tasks = dict(adapter.tasks())
    ar_001 = tasks["ar_001"]
    assert "Elena" in ar_001.description
    assert "mother's name" in ar_001.description


# ---------- verify() — substring match ----------


def _make_trajectory(answer: str | None) -> Trajectory:
    final_state = {"answer": answer} if answer is not None else None
    return Trajectory(
        goal_id=uuid4(),
        status=TrajectoryStatus.SUCCESS,
        final_state=final_state,
        total_cost=Cost(),
    )


def test_verify_substring_match_pass(adapter: MemoryAgentBenchAdapter) -> None:
    traj = _make_trajectory("Her name is Elena, she lives in Saratov.")
    ok, mode = adapter.verify("ar_001", traj)
    assert ok is True
    assert mode is None


def test_verify_substring_match_case_insensitive(
    adapter: MemoryAgentBenchAdapter,
) -> None:
    traj = _make_trajectory("PIXEL is the dog's name")
    ok, _ = adapter.verify("ar_002", traj)
    assert ok is True


def test_verify_substring_match_fail(adapter: MemoryAgentBenchAdapter) -> None:
    traj = _make_trajectory("Bobby")  # wrong name
    ok, mode = adapter.verify("ar_002", traj)
    assert ok is False
    assert mode == "substring_not_found"


def test_verify_handles_no_final_state(adapter: MemoryAgentBenchAdapter) -> None:
    traj = _make_trajectory(None)
    ok, mode = adapter.verify("ar_001", traj)
    assert ok is False
    assert mode == "no_final_state"


def test_verify_handles_empty_answer(adapter: MemoryAgentBenchAdapter) -> None:
    traj = _make_trajectory("")
    ok, mode = adapter.verify("ar_001", traj)
    assert ok is False
    assert mode == "empty_answer"


def test_verify_unknown_task_id(adapter: MemoryAgentBenchAdapter) -> None:
    traj = _make_trajectory("anything")
    ok, mode = adapter.verify("nonexistent", traj)
    assert ok is False
    assert mode == "unknown_task"


def test_verify_handles_response_field_fallback() -> None:
    """final_state может содержать 'response' вместо 'answer' — fallback."""
    a = MemoryAgentBenchAdapter(
        tasks=[
            {"task_id": "x", "context": "", "question": "?", "expected_answer": "hi"}
        ]
    )
    traj = Trajectory(
        goal_id=uuid4(),
        status=TrajectoryStatus.SUCCESS,
        final_state={"response": "hello hi there"},
    )
    ok, _ = a.verify("x", traj)
    assert ok is True


# ---------- Integration with run_evaluation ----------


def test_run_evaluation_with_adapter(adapter: MemoryAgentBenchAdapter) -> None:
    """Полный прогон через harness — mock-agent отвечает разное."""
    # Заранее ответы по task_id; в реальном agent они бы пришли из ReAct
    answers = {
        "ar_001": "Her name is Elena",  # correct
        "ar_002": "It is Pixel",  # correct
        "cr_001": "Acme Corp",  # WRONG — should be Globex
        "ttl_001": "unlock door, of course",  # correct
        "lru_001": "Bob wrote it",  # WRONG — should be Alice
    }

    def mock_agent(goal):
        task_id = goal.metadata["task_id"]
        ans = answers.get(task_id, "")
        return Trajectory(
            goal_id=goal.id,
            status=TrajectoryStatus.SUCCESS,
            final_state={"answer": ans},
            total_cost=Cost(tokens=50),
        )

    result = run_evaluation(adapter, mock_agent)

    assert result.name == "memory_agent_bench_sample"
    assert len(result.per_task) == 5
    assert result.success_rate == pytest.approx(3 / 5)
    assert "substring_not_found" in result.failure_modes
    assert result.failure_modes["substring_not_found"] == 2


def test_run_evaluation_limit_applies(adapter: MemoryAgentBenchAdapter) -> None:
    def trivial_agent(goal):
        return Trajectory(
            goal_id=goal.id,
            status=TrajectoryStatus.SUCCESS,
            final_state={"answer": goal.metadata["expected_answer"]},
            total_cost=Cost(),
        )

    result = run_evaluation(adapter, trivial_agent, limit=2)
    assert len(result.per_task) == 2
    assert result.success_rate == 1.0
