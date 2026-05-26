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
) -> EvalResult:
    """Главный entry-point harness'а.

    Параметры:
    - adapter: BenchmarkAdapter — поставщик задач + верификатор
    - agent_run: callable(goal) → Trajectory — наш агент в режиме «выполнить goal»
    - limit: ограничение числа задач (None = all)
    """
    result = EvalResult(name=adapter.name)
    for i, (task_id, goal) in enumerate(adapter.tasks()):
        if limit is not None and i >= limit:
            break
        log.info("eval.task.start", adapter=adapter.name, task_id=task_id)
        try:
            traj = agent_run(goal)
        except Exception as exc:  # noqa: BLE001
            log.error("eval.task.crashed", task_id=task_id, error=str(exc))
            result.per_task.append(
                PerTaskResult(
                    task_id=task_id,
                    success=False,
                    failure_mode=f"agent_crash:{type(exc).__name__}",
                )
            )
            continue

        success, failure_mode = adapter.verify(task_id, traj)
        per = PerTaskResult(
            task_id=task_id,
            success=success,
            trajectory=traj,
            failure_mode=failure_mode,
            cost_tokens=traj.total_cost.tokens,
            steps=len(traj.steps),
        )
        result.per_task.append(per)
        log.info(
            "eval.task.done",
            adapter=adapter.name,
            task_id=task_id,
            success=success,
            failure_mode=failure_mode,
        )
    log.info(
        "eval.run.done",
        adapter=adapter.name,
        tasks=len(result.per_task),
        success_rate=result.success_rate,
    )
    return result
