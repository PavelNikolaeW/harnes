"""Tests for harnes.eval.history."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harnes.eval import (
    EvalHistoryStore,
    EvalResult,
    MemoryAgentBenchAdapter,
    PerTaskResult,
    run_evaluation,
)
from harnes.eval.history import _capture_config_hash, capture_skill_versions
from harnes.react.schema import Cost, Trajectory, TrajectoryStatus
from harnes.skills.schema import Skill
from harnes.skills.store import SkillRegistry


# ---------- helpers ----------


def _eval_result(name: str, n_success: int, n_fail: int) -> EvalResult:
    result = EvalResult(name=name)
    for i in range(n_success):
        result.per_task.append(
            PerTaskResult(task_id=f"s{i}", success=True, cost_tokens=100, steps=2)
        )
    for i in range(n_fail):
        result.per_task.append(
            PerTaskResult(
                task_id=f"f{i}",
                success=False,
                cost_tokens=200,
                steps=5,
                failure_mode="substring_not_found",
            )
        )
    return result


# ---------- Store basics ----------


def test_history_store_record_and_get() -> None:
    store = EvalHistoryStore(":memory:")
    result = _eval_result("test_bench", n_success=3, n_fail=2)
    now = datetime.now(UTC)

    row = store.record(
        result=result,
        started_at=now,
        ended_at=now + timedelta(seconds=10),
        skill_versions={"general": "0.0.1"},
        notes="baseline",
    )
    assert row.id is not None
    assert row.total_tasks == 5
    assert row.success_count == 3
    assert row.success_rate == 0.6
    assert row.notes == "baseline"
    assert json.loads(row.skill_versions_json) == {"general": "0.0.1"}
    assert "substring_not_found" in row.failure_modes_json

    fetched = store.get(row.id)
    assert fetched is not None
    assert fetched.success_rate == 0.6


def test_history_store_list_and_latest() -> None:
    store = EvalHistoryStore(":memory:")
    now = datetime.now(UTC)
    for i in range(3):
        store.record(
            result=_eval_result("adapter_a", n_success=i, n_fail=5 - i),
            started_at=now,
            ended_at=now,
        )
    store.record(
        result=_eval_result("adapter_b", n_success=10, n_fail=0),
        started_at=now,
        ended_at=now,
    )

    all_runs = store.list_runs()
    assert len(all_runs) == 4

    a_runs = store.list_runs(adapter_name="adapter_a")
    assert len(a_runs) == 3
    assert all(r.adapter_name == "adapter_a" for r in a_runs)

    latest_a = store.latest(adapter_name="adapter_a")
    assert latest_a is not None
    assert latest_a.success_count == 2  # the last one created

    latest_b = store.latest(adapter_name="adapter_b")
    assert latest_b is not None
    assert latest_b.adapter_name == "adapter_b"


def test_history_store_persists_to_file(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    store1 = EvalHistoryStore(db_path)
    now = datetime.now(UTC)
    store1.record(
        result=_eval_result("x", 1, 0),
        started_at=now,
        ended_at=now,
    )

    store2 = EvalHistoryStore(db_path)
    runs = store2.list_runs()
    assert len(runs) == 1
    assert runs[0].adapter_name == "x"


# ---------- Config hash ----------


def test_config_hash_stable() -> None:
    h1 = _capture_config_hash()
    h2 = _capture_config_hash()
    assert h1 == h2
    assert len(h1) == 16


# ---------- Skill versions snapshot ----------


def test_capture_skill_versions(tmp_path: Path) -> None:
    reg = SkillRegistry(bundles_dir=tmp_path, metrics_db=":memory:")
    reg.save(
        Skill(
            id="general",
            name="general",
            description="g",
            version="0.0.1",
            prompt_template="x",
        )
    )
    reg.save(
        Skill(
            id="other",
            name="other",
            description="o",
            version="1.0.0",
            prompt_template="y",
        )
    )
    versions = capture_skill_versions(reg)
    assert versions == {"general": "0.0.1", "other": "1.0.0"}


def test_capture_skill_versions_handles_broken_registry() -> None:
    """Если registry бросит — пустой dict, не падает."""

    class Broken:
        def load_all(self):
            raise RuntimeError("nope")

    assert capture_skill_versions(Broken()) == {}


# ---------- run_evaluation integration ----------


def test_run_evaluation_writes_to_history(tmp_path: Path) -> None:
    fixture = (
        Path(__file__).resolve().parent / "fixtures" / "memory_agent_bench_sample.json"
    )
    adapter = MemoryAgentBenchAdapter(tasks_file=fixture)

    history = EvalHistoryStore(tmp_path / "h.db")

    def perfect_agent(goal):
        return Trajectory(
            goal_id=goal.id,
            status=TrajectoryStatus.SUCCESS,
            final_state={"answer": goal.metadata["expected_answer"]},
            total_cost=Cost(tokens=50),
        )

    result = run_evaluation(
        adapter, perfect_agent, history_repo=history, notes="test-run"
    )
    assert result.success_rate == 1.0

    runs = history.list_runs()
    assert len(runs) == 1
    assert runs[0].adapter_name == "memory_agent_bench_sample"
    assert runs[0].success_rate == 1.0
    assert runs[0].notes == "test-run"


def test_run_evaluation_without_history_repo_is_noop_for_history() -> None:
    fixture = (
        Path(__file__).resolve().parent / "fixtures" / "memory_agent_bench_sample.json"
    )
    adapter = MemoryAgentBenchAdapter(tasks_file=fixture)

    def trivial(goal):
        return Trajectory(
            goal_id=goal.id,
            status=TrajectoryStatus.SUCCESS,
            final_state={"answer": "x"},
            total_cost=Cost(),
        )

    # history_repo=None — никаких записей не делается, exception не бросается
    result = run_evaluation(adapter, trivial)
    assert result is not None
