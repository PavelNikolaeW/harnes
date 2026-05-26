"""MemoryAgentBench adapter (ICLR 2026).

Источник: https://github.com/HUST-AI-HYZ/MemoryAgentBench
Датасет: huggingface.co/datasets/ai-hyz/MemoryAgentBench

В v0.2 — minimal scaffolding adapter:
- Загружает задачи из локального JSON (committed sample) или внешнего файла
- Каждый task превращается в наш Goal: context+question → description,
  с structural-predicate и судьёй на substring_exact_match
- Verify (наша часть Protocol'а) делает substring_exact_match: ответ агента
  должен содержать `expected_answer` как подстроку (case-insensitive)

Метрики MemoryAgentBench:
- substring_exact_match: для AR (event_qa, ruler_qa*), CR (fact_mh, fact_sh)
- exact_match: для LRU (detectiveQA), TTL (ICL_*)

В v0.2 поддерживаем только substring_exact_match. exact_match добавим
при подключении реальных HF-датасетов в v0.3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import structlog

from harnes.eval.harness import BenchmarkAdapter
from harnes.goals.schema import (
    Goal,
    GoalClass,
    JudgePredicate,
    Origin,
)
from harnes.react.schema import Trajectory

log = structlog.get_logger()


class MemoryAgentBenchAdapter(BenchmarkAdapter):
    """Adapter для MemoryAgentBench-format задач.

    Конструктор принимает либо путь к JSON-файлу с задачами, либо сам список
    задач (для тестов).

    Формат JSON:
    {
      "name": "...",
      "metric": "substring_exact_match" | "exact_match",
      "tasks": [
        {
          "task_id": "...",
          "category": "accurate_retrieval" | "conflict_resolution" | ...,
          "context": "<long-text injection>",
          "question": "<query>",
          "expected_answer": "<substring or exact match>"
        }, ...
      ]
    }
    """

    name = "memory_agent_bench"

    def __init__(
        self,
        tasks_file: Path | str | None = None,
        tasks: list[dict[str, Any]] | None = None,
        metric: str = "substring_exact_match",
    ) -> None:
        if tasks is not None:
            self._tasks = tasks
            self.metric = metric
        elif tasks_file is not None:
            with Path(tasks_file).open(encoding="utf-8") as f:
                data = json.load(f)
            self._tasks = data.get("tasks", [])
            self.metric = data.get("metric", metric)
            self.name = data.get("name", self.name)
        else:
            raise ValueError("Either tasks_file or tasks must be provided")

    # ---------- BenchmarkAdapter Protocol ----------

    def tasks(self) -> Iterable[tuple[str, Goal]]:
        for t in self._tasks:
            task_id = str(t["task_id"])
            context = str(t.get("context", ""))
            question = str(t["question"])
            expected = str(t["expected_answer"])

            # Goal description содержит context + question.
            # Агент должен выработать ответ в trajectory.final_state.answer.
            description = (
                f"Memory task [{t.get('category', '?')}]. "
                f"Context (remember this):\n{context}\n\n"
                f"Question: {question}\n\n"
                "Respond by calling tool_id=finish with "
                'args={"final_state": {"answer": "<your concise answer>"}}.'
            )

            goal = Goal(
                description=description,
                goal_class=GoalClass.TASK,
                predicate_of_success=JudgePredicate(
                    criterion=(
                        f"trajectory.final_state.answer contains the expected value "
                        f"{expected!r} (case-insensitive substring)"
                    )
                ),
                origin=Origin.OPERATOR,
                originator=f"benchmark:{self.name}",
                metadata={
                    "benchmark": self.name,
                    "task_id": task_id,
                    "expected_answer": expected,
                    "category": t.get("category", ""),
                    "metric": self.metric,
                },
            )
            yield task_id, goal

    def verify(self, task_id: str, trajectory: Trajectory) -> tuple[bool, str | None]:
        """Сверяет trajectory.final_state с expected_answer по выбранной метрике."""
        # Найти task по id для expected_answer.
        task = next((t for t in self._tasks if str(t["task_id"]) == task_id), None)
        if task is None:
            return False, "unknown_task"

        expected = str(task["expected_answer"])
        metric = task.get("metric", self.metric)

        # Извлечь ответ из final_state.
        fs = trajectory.final_state
        if fs is None:
            return False, "no_final_state"
        if isinstance(fs, dict):
            answer = str(fs.get("answer", "") or fs.get("response", "") or "")
        else:
            answer = str(fs)
        if not answer:
            return False, "empty_answer"

        if metric == "substring_exact_match":
            ok = expected.lower() in answer.lower()
            return ok, None if ok else "substring_not_found"
        if metric == "exact_match":
            ok = answer.strip().lower() == expected.strip().lower()
            return ok, None if ok else "exact_mismatch"

        log.warning("memory_agent_bench.unknown_metric", metric=metric)
        return False, f"unknown_metric:{metric}"
