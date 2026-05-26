"""Eval harness scaffolding.

Задача:
- определить типы PerTaskResult / EvalResult, через которые отчитывается прогон;
- BenchmarkAdapter — контракт, который реализует интеграция с конкретным
  бенчмарком (MemoryAgentBench и любым другим);
- run_evaluation — оркестратор: для каждой задачи adapter превращает её в Goal,
  агент исполняет, adapter сверяет результат → PerTaskResult.

В v0 включён только StubAdapter (для тестов) — он не дёргает реальный бенч.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import structlog

from harnes.goals.schema import Goal
from harnes.react.schema import Trajectory

log = structlog.get_logger()


# ---------- Result types ----------


@dataclass
class PerTaskResult:
    task_id: str
    success: bool
    trajectory: Trajectory | None = None
    failure_mode: str | None = None
    cost_tokens: int = 0
    steps: int = 0
    latency_s: float = 0.0  # v1.0 #32: wall-clock на эту попытку
    attempt: int = 0  # v1.0 #32: индекс попытки (0..repeat_k-1)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    name: str
    per_task: list[PerTaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.per_task:
            return 0.0
        return sum(1 for r in self.per_task if r.success) / len(self.per_task)

    @property
    def avg_cost_tokens(self) -> float:
        if not self.per_task:
            return 0.0
        return sum(r.cost_tokens for r in self.per_task) / len(self.per_task)

    @property
    def avg_steps(self) -> float:
        if not self.per_task:
            return 0.0
        return sum(r.steps for r in self.per_task) / len(self.per_task)

    @property
    def failure_modes(self) -> dict[str, int]:
        modes: dict[str, int] = {}
        for r in self.per_task:
            if not r.success and r.failure_mode:
                modes[r.failure_mode] = modes.get(r.failure_mode, 0) + 1
        return modes


# ---------- Adapter contract ----------


class BenchmarkAdapter(Protocol):
    """Контракт интеграции с конкретным бенчмарком.

    Любой бенч (MemoryAgentBench, свой, синтетический) реализует этот Protocol —
    дальше run_evaluation работает с ним абстрактно.
    """

    name: str

    def tasks(self) -> Iterable[tuple[str, Goal]]:
        """Yield (task_id, Goal) — каждая задача переведена в нашу Goal-схему."""
        ...

    def verify(self, task_id: str, trajectory: Trajectory) -> tuple[bool, str | None]:
        """Своя верификация по схеме бенча. Возвращает (success, failure_mode).

        Используется поверх нашего стандартного verify, потому что у бенча
        могут быть свои критерии и dataset-specific проверки.
        """
        ...


# ---------- Orchestrator ----------


def run_evaluation(
    adapter: BenchmarkAdapter,
    agent_run: Callable[[Goal], Trajectory],
    limit: int | None = None,
    history_repo: Any = None,
    skill_registry: Any = None,
    notes: str = "",
    *,
    eval_set: str = "",
    held_out: bool = False,
    repeat_k: int = 1,
) -> EvalResult:
    """Главный entry-point harness'а.

    Параметры:
    - adapter: BenchmarkAdapter — поставщик задач + верификатор
    - agent_run: callable(goal) → Trajectory — наш агент в режиме «выполнить goal»
    - limit: ограничение числа задач (None = all)
    - history_repo: опционально — EvalHistoryStore; если передан, прогон
      сохраняется в SQLite с git_sha и skill_versions snapshot
    - skill_registry: опционально — для snapshot'а skill-версий в history
    - notes: свободный текст для записи в history (например, «baseline после #26»)
    - eval_set: v1.0 #31 — короткий идентификатор набора задач
    - held_out: v1.0 #31 — флаг held-out (не смотреть при разработке)
    - repeat_k: v1.0 #32 — повторить каждую task k раз (pass@k / stable@k)
    """
    import time
    from datetime import UTC, datetime

    started_at = datetime.now(UTC)

    result = EvalResult(name=adapter.name)
    # Материализуем задачи один раз — иначе при k>1 нельзя их пройти повторно.
    all_tasks = list(adapter.tasks())
    if limit is not None:
        all_tasks = all_tasks[:limit]

    for task_id, goal in all_tasks:
        for attempt in range(repeat_k):
            log.info(
                "eval.task.start",
                adapter=adapter.name,
                task_id=task_id,
                attempt=attempt,
                repeat_k=repeat_k,
            )
            t0 = time.monotonic()
            try:
                traj = agent_run(goal)
            except Exception as exc:  # noqa: BLE001
                log.error("eval.task.crashed", task_id=task_id, error=str(exc))
                result.per_task.append(
                    PerTaskResult(
                        task_id=task_id,
                        success=False,
                        failure_mode=f"agent_crash:{type(exc).__name__}",
                        latency_s=time.monotonic() - t0,
                        attempt=attempt,
                    )
                )
                continue
            latency = time.monotonic() - t0

            success, failure_mode = adapter.verify(task_id, traj)
            per = PerTaskResult(
                task_id=task_id,
                success=success,
                trajectory=traj,
                failure_mode=failure_mode,
                cost_tokens=traj.total_cost.tokens,
                steps=len(traj.steps),
                latency_s=latency,
                attempt=attempt,
            )
            result.per_task.append(per)
            log.info(
                "eval.task.done",
                adapter=adapter.name,
                task_id=task_id,
                attempt=attempt,
                success=success,
                failure_mode=failure_mode,
            )

    ended_at = datetime.now(UTC)

    log.info(
        "eval.run.done",
        adapter=adapter.name,
        tasks=len(result.per_task),
        unique_tasks=len({r.task_id for r in result.per_task}),
        repeat_k=repeat_k,
        success_rate=result.success_rate,
    )

    # Persist в history если запрошено.
    if history_repo is not None:
        from harnes.eval.history import capture_skill_versions

        skill_versions: dict[str, str] = {}
        if skill_registry is not None:
            skill_versions = capture_skill_versions(skill_registry)
        history_repo.record(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            skill_versions=skill_versions,
            notes=notes,
            eval_set=eval_set,
            held_out=held_out,
            repeat_k=repeat_k,
        )

    return result
