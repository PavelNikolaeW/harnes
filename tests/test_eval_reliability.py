"""Tests for v1.0 #31+#32: held-out + reliability metrics."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from harnes.eval import (
    EvalHistoryStore,
    EvalResult,
    PerTaskResult,
    run_evaluation,
)
from harnes.eval.history import (
    _quantile,
    _shannon_entropy,
    compute_eval_set_hash,
    compute_reliability_metrics,
)
from harnes.goals.schema import Goal, GoalClass, JudgePredicate, Origin
from harnes.react.schema import Cost, Trajectory, TrajectoryStatus


# ---------- _quantile ----------


def test_quantile_empty_zero() -> None:
    assert _quantile([], 0.5) == 0.0


def test_quantile_single_returns_value() -> None:
    assert _quantile([3.0], 0.5) == 3.0
    assert _quantile([3.0], 0.95) == 3.0


def test_quantile_basic() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _quantile(data, 0.0) == 1.0
    assert _quantile(data, 0.5) == 3.0
    assert _quantile(data, 1.0) == 5.0


def test_quantile_p95_interpolated() -> None:
    data = list(range(1, 21))  # 1..20
    # 0.95 * 19 = 18.05 → между 19.0 и 20.0
    v = _quantile(data, 0.95)
    assert abs(v - 19.05) < 1e-9


# ---------- _shannon_entropy ----------


def test_entropy_empty() -> None:
    assert _shannon_entropy({}) == 0.0


def test_entropy_single_mode() -> None:
    """Один доминирующий режим — энтропия 0."""
    assert _shannon_entropy({"only_mode": 10}) == 0.0


def test_entropy_uniform_two_modes() -> None:
    """Два равных режима — энтропия log2(2) = 1."""
    h = _shannon_entropy({"a": 5, "b": 5})
    assert abs(h - 1.0) < 1e-9


def test_entropy_uniform_four_modes() -> None:
    h = _shannon_entropy({"a": 1, "b": 1, "c": 1, "d": 1})
    assert abs(h - 2.0) < 1e-9


def test_entropy_skewed_lower_than_uniform() -> None:
    h_uniform = _shannon_entropy({"a": 5, "b": 5})
    h_skewed = _shannon_entropy({"a": 9, "b": 1})
    assert h_skewed < h_uniform


# ---------- compute_eval_set_hash ----------


def test_eval_set_hash_stable_under_reorder() -> None:
    h1 = compute_eval_set_hash(["b", "a", "c"])
    h2 = compute_eval_set_hash(["c", "a", "b"])
    assert h1 == h2


def test_eval_set_hash_differs_for_different_sets() -> None:
    assert compute_eval_set_hash(["a", "b"]) != compute_eval_set_hash(["a", "c"])


def test_eval_set_hash_length() -> None:
    h = compute_eval_set_hash(["x"])
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ---------- compute_reliability_metrics ----------


def _make_per_task(
    task_id: str, success: bool, steps: int, latency: float, attempt: int = 0,
    answer: str = "x", failure_mode: str | None = None,
) -> PerTaskResult:
    traj = Trajectory(
        goal_id=Goal(
            description="t",
            goal_class=GoalClass.TASK,
            predicate_of_success=JudgePredicate(criterion="x"),
            origin=Origin.OPERATOR,
            originator="t",
        ).id,
        status=TrajectoryStatus.SUCCESS if success else TrajectoryStatus.FAILURE,
        final_state={"answer": answer},
        total_cost=Cost(),
    )
    return PerTaskResult(
        task_id=task_id,
        success=success,
        trajectory=traj,
        failure_mode=failure_mode if not success else None,
        steps=steps,
        latency_s=latency,
        attempt=attempt,
    )


def test_metrics_empty() -> None:
    m = compute_reliability_metrics([], repeat_k=1)
    assert m["pass_at_k"] == 0.0
    assert m["stable_at_k"] == 0.0
    assert m["failure_entropy"] == 0.0


def test_metrics_pass_at_k_1_equals_success_rate() -> None:
    pt = [
        _make_per_task("t0", True, 3, 1.0),
        _make_per_task("t1", False, 5, 2.0, failure_mode="x"),
        _make_per_task("t2", True, 2, 1.5),
    ]
    m = compute_reliability_metrics(pt, repeat_k=1)
    assert m["pass_at_k"] == pytest.approx(2 / 3)


def test_metrics_stable_at_k_1_is_one() -> None:
    """При k=1 каждая task тривиально стабильна."""
    pt = [_make_per_task("t0", True, 3, 1.0, answer="a")]
    m = compute_reliability_metrics(pt, repeat_k=1)
    assert m["stable_at_k"] == 1.0


def test_metrics_pass_at_k_with_repeats() -> None:
    """Task с k=2: если хоть один успешен — она зачитывается."""
    pt = [
        # t0: оба провалились — не зачитывается
        _make_per_task("t0", False, 5, 1.0, attempt=0, failure_mode="x"),
        _make_per_task("t0", False, 5, 1.0, attempt=1, failure_mode="x"),
        # t1: первый провал, второй ОК — зачитывается
        _make_per_task("t1", False, 5, 1.0, attempt=0, failure_mode="x"),
        _make_per_task("t1", True, 3, 1.0, attempt=1, answer="ok"),
        # t2: оба успешны
        _make_per_task("t2", True, 2, 1.0, attempt=0, answer="yes"),
        _make_per_task("t2", True, 2, 1.0, attempt=1, answer="yes"),
    ]
    m = compute_reliability_metrics(pt, repeat_k=2)
    # 2 из 3 task'ов прошли (t1, t2)
    assert m["pass_at_k"] == pytest.approx(2 / 3)


def test_metrics_stable_at_k_detects_inconsistency() -> None:
    """Task стабильна если все k запусков дали один canonical answer."""
    pt = [
        # t0: один и тот же ответ — стабильна
        _make_per_task("t0", True, 3, 1.0, attempt=0, answer="ok"),
        _make_per_task("t0", True, 3, 1.0, attempt=1, answer="OK"),  # case-insensitive
        # t1: разные ответы — нестабильна
        _make_per_task("t1", True, 3, 1.0, attempt=0, answer="alpha"),
        _make_per_task("t1", True, 3, 1.0, attempt=1, answer="beta"),
    ]
    m = compute_reliability_metrics(pt, repeat_k=2)
    assert m["stable_at_k"] == pytest.approx(1 / 2)


def test_metrics_quantiles_steps_and_latency() -> None:
    pt = [
        _make_per_task(f"t{i}", True, steps=i + 1, latency=float(i + 1))
        for i in range(10)
    ]
    m = compute_reliability_metrics(pt, repeat_k=1)
    # 0..9 reordered → 1..10, median 5.5
    assert m["p50_steps"] == pytest.approx(5.5)
    assert m["p50_latency_s"] == pytest.approx(5.5)
    # p95 of 1..10 → 0.95*9=8.55 → 9.55
    assert m["p95_steps"] == pytest.approx(9.55, abs=1e-9)


def test_metrics_failure_entropy_collapsed() -> None:
    """Все провалы — один режим, entropy = 0."""
    pt = [
        _make_per_task(f"t{i}", False, 5, 1.0, failure_mode="same")
        for i in range(5)
    ]
    m = compute_reliability_metrics(pt, repeat_k=1)
    assert m["failure_entropy"] == 0.0


# ---------- EvalHistoryStore.record with v1.0 fields ----------


def _eval_result(name: str = "x", n: int = 3) -> EvalResult:
    """Maker с per_task для пары рантов в одну функцию."""
    r = EvalResult(name=name)
    for i in range(n):
        r.per_task.append(
            PerTaskResult(task_id=f"t{i}", success=i % 2 == 0, steps=2, cost_tokens=100)
        )
    return r


def test_record_writes_v1_fields(tmp_path: Path) -> None:
    store = EvalHistoryStore(tmp_path / "h.db")
    now = datetime.now(UTC)

    row = store.record(
        result=_eval_result("test_bench", n=4),
        started_at=now,
        ended_at=now + timedelta(seconds=1),
        eval_set="mab_holdout_lru_v1",
        held_out=True,
        repeat_k=2,
    )
    assert row.eval_set == "mab_holdout_lru_v1"
    assert row.held_out is True
    assert row.repeat_k == 2
    # 4 unique task_ids (t0..t3); pass@k посчитан из per_task
    assert 0 <= row.pass_at_k <= 1
    assert 0 <= row.stable_at_k <= 1
    assert row.eval_set_hash  # хэш вычислен


def test_list_runs_hides_held_out_by_default(tmp_path: Path) -> None:
    store = EvalHistoryStore(tmp_path / "h.db")
    now = datetime.now(UTC)
    # 1 dev + 1 held-out
    store.record(
        result=_eval_result("x"), started_at=now, ended_at=now,
        eval_set="dev", held_out=False,
    )
    store.record(
        result=_eval_result("x"), started_at=now, ended_at=now,
        eval_set="holdout", held_out=True,
    )

    visible = store.list_runs()
    assert len(visible) == 1
    assert visible[0].held_out is False

    all_runs = store.list_runs(include_held_out=True)
    assert len(all_runs) == 2


def test_list_runs_filter_by_eval_set(tmp_path: Path) -> None:
    store = EvalHistoryStore(tmp_path / "h.db")
    now = datetime.now(UTC)
    store.record(
        result=_eval_result("x"), started_at=now, ended_at=now, eval_set="dev_ar",
    )
    store.record(
        result=_eval_result("x"), started_at=now, ended_at=now, eval_set="dev_ttl",
    )

    ar = store.list_runs(eval_set="dev_ar")
    assert len(ar) == 1
    assert ar[0].eval_set == "dev_ar"


def test_latest_respects_held_out_filter(tmp_path: Path) -> None:
    store = EvalHistoryStore(tmp_path / "h.db")
    now = datetime.now(UTC)
    store.record(
        result=_eval_result("x"), started_at=now, ended_at=now, held_out=False,
    )
    store.record(
        result=_eval_result("x"), started_at=now, ended_at=now, held_out=True,
    )
    # default — dev
    latest_dev = store.latest(adapter_name="x")
    assert latest_dev is not None
    assert latest_dev.held_out is False
    # include_held_out=True — берёт самый последний (held-out)
    latest_any = store.latest(adapter_name="x", include_held_out=True)
    assert latest_any is not None
    assert latest_any.held_out is True


# ---------- Миграция: старая БД без v1.0 колонок ----------


def test_migration_adds_columns_to_legacy_db(tmp_path: Path) -> None:
    """Создаём «старую» БД с pre-v1.0 схемой, открываем v1.0 store — должно мигрировать."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    # Минимальная старая схема (без новых колонок).
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            adapter_name TEXT NOT NULL,
            started_at DATETIME NOT NULL,
            ended_at DATETIME NOT NULL,
            total_tasks INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            success_rate REAL NOT NULL DEFAULT 0.0,
            avg_steps REAL NOT NULL DEFAULT 0.0,
            avg_cost_tokens REAL NOT NULL DEFAULT 0.0,
            failure_modes_json TEXT NOT NULL DEFAULT '{}',
            git_sha TEXT NOT NULL DEFAULT '',
            skill_versions_json TEXT NOT NULL DEFAULT '{}',
            config_hash TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT ''
        );
        INSERT INTO eval_runs (
            adapter_name, started_at, ended_at, total_tasks, success_count, success_rate
        ) VALUES ('old_run', '2024-01-01', '2024-01-01', 5, 3, 0.6);
        """
    )
    conn.commit()
    conn.close()

    # Открываем через v1.0 store — миграция должна сработать.
    store = EvalHistoryStore(db_path)
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].adapter_name == "old_run"
    assert runs[0].success_rate == 0.6
    # Новые поля — с дефолтами.
    assert runs[0].held_out is False
    assert runs[0].repeat_k == 1
    assert runs[0].pass_at_k == 0.0
    assert runs[0].eval_set == ""

    # И запись новых данных должна работать.
    store.record(
        result=_eval_result("new"),
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        eval_set="dev",
        held_out=False,
    )
    rows = store.list_runs(adapter_name="new")
    assert len(rows) == 1


# ---------- run_evaluation с repeat_k ----------


class _StubAdapter:
    name = "stub"

    def __init__(self, task_ids: list[str]) -> None:
        self._task_ids = task_ids

    def tasks(self):
        for tid in self._task_ids:
            yield tid, Goal(
                description=tid,
                goal_class=GoalClass.TASK,
                predicate_of_success=JudgePredicate(criterion="x"),
                origin=Origin.OPERATOR,
                originator="stub",
                metadata={"task_id": tid},
            )

    def verify(self, task_id, trajectory):
        # Успех если trajectory.final_state.answer == "ok"
        if isinstance(trajectory.final_state, dict):
            ok = trajectory.final_state.get("answer") == "ok"
        else:
            ok = False
        return ok, None if ok else "wrong_answer"


def test_run_evaluation_repeat_k_runs_each_task_k_times() -> None:
    calls: dict[str, int] = {}

    def agent_run(goal: Goal) -> Trajectory:
        tid = goal.metadata.get("task_id")
        calls[tid] = calls.get(tid, 0) + 1
        return Trajectory(
            goal_id=goal.id,
            status=TrajectoryStatus.SUCCESS,
            final_state={"answer": "ok"},
            total_cost=Cost(tokens=10),
        )

    adapter = _StubAdapter(["t1", "t2"])
    result = run_evaluation(adapter, agent_run, repeat_k=3)

    assert calls == {"t1": 3, "t2": 3}
    assert len(result.per_task) == 6
    attempts = sorted([(r.task_id, r.attempt) for r in result.per_task])
    assert attempts == [
        ("t1", 0), ("t1", 1), ("t1", 2),
        ("t2", 0), ("t2", 1), ("t2", 2),
    ]


def test_run_evaluation_repeat_k_records_pass_at_k(tmp_path: Path) -> None:
    """Один из k запусков успешен → task зачитывается в pass@k."""
    counter = {"t0": 0}

    def flaky_agent(goal: Goal) -> Trajectory:
        # task t0: первый запуск проваливается, второй — ОК.
        counter["t0"] += 1
        answer = "ok" if counter["t0"] >= 2 else "wrong"
        return Trajectory(
            goal_id=goal.id,
            status=TrajectoryStatus.SUCCESS,
            final_state={"answer": answer},
            total_cost=Cost(tokens=10),
        )

    adapter = _StubAdapter(["t0"])
    history = EvalHistoryStore(tmp_path / "h.db")

    result = run_evaluation(
        adapter, flaky_agent, history_repo=history, repeat_k=2, eval_set="dev",
    )

    # success_rate over всех попыток = 1/2 = 50%, но pass@2 = 100%.
    assert result.success_rate == pytest.approx(0.5)
    runs = history.list_runs()
    assert len(runs) == 1
    assert runs[0].pass_at_k == pytest.approx(1.0)
    assert runs[0].repeat_k == 2


def test_run_evaluation_latency_recorded() -> None:
    def slow_agent(goal: Goal) -> Trajectory:
        import time
        time.sleep(0.01)
        return Trajectory(
            goal_id=goal.id,
            status=TrajectoryStatus.SUCCESS,
            final_state={"answer": "ok"},
            total_cost=Cost(tokens=1),
        )

    adapter = _StubAdapter(["t1"])
    result = run_evaluation(adapter, slow_agent, repeat_k=1)
    assert result.per_task[0].latency_s > 0.005


def test_run_evaluation_records_held_out_flag(tmp_path: Path) -> None:
    def trivial(goal: Goal) -> Trajectory:
        return Trajectory(
            goal_id=goal.id,
            status=TrajectoryStatus.SUCCESS,
            final_state={"answer": "ok"},
            total_cost=Cost(tokens=1),
        )

    adapter = _StubAdapter(["t1"])
    history = EvalHistoryStore(tmp_path / "h.db")
    run_evaluation(
        adapter, trivial,
        history_repo=history,
        eval_set="mab_holdout_lru_v1",
        held_out=True,
    )

    # default list — пусто (скрывает held-out)
    assert store_count(history, include_held_out=False) == 0
    # с флагом — есть
    rows = history.list_runs(include_held_out=True)
    assert len(rows) == 1
    assert rows[0].held_out is True
    assert rows[0].eval_set == "mab_holdout_lru_v1"


def store_count(store: EvalHistoryStore, include_held_out: bool) -> int:
    return len(store.list_runs(include_held_out=include_held_out, limit=100))
