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


# HF dataset на huggingface.co/datasets/ai-hyz/MemoryAgentBench
HF_DATASET_ID = "ai-hyz/MemoryAgentBench"

# Splits в HF-датасете → наша category-нотация
_HF_SPLITS = {
    "Accurate_Retrieval": "accurate_retrieval",
    "Test_Time_Learning": "test_time_learning",
    "Long_Range_Understanding": "long_range_understanding",
    "Conflict_Resolution": "conflict_resolution",
}

# Метрика по категории, как у авторов в README.md (см. § "Clarification on Eval Metrics")
_CATEGORY_METRICS = {
    "accurate_retrieval": "substring_exact_match",
    "conflict_resolution": "substring_exact_match",
    "test_time_learning": "exact_match",
    "long_range_understanding": "exact_match",
}


def load_hf_tasks(
    splits: list[str] | None = None,
    limit_examples_per_split: int | None = None,
    limit_questions_per_example: int | None = None,
    chunk_threshold_chars: int | None = 4000,
    chunk_max_chars: int = 2000,
) -> list[dict[str, Any]]:
    """Грузит задачи из HF ai-hyz/MemoryAgentBench и расплющивает в наш формат.

    Каждая HF-строка содержит один context + список вопросов (до 100) + список
    acceptable-answers (multi-valid). Flatten: каждый (context, q, answers)
    становится отдельным task'ом в нашем JSON-формате.

    Multi-turn injection: если len(context) > chunk_threshold_chars, контекст
    нарезается на chunks (chunk_max_chars символов на chunk) и сохраняется
    в task['chunks']. Поле task['context'] оставляется коротким (или пустым).
    При chunk_threshold_chars=None — chunking отключён (всё в context).

    Parameters
    ----------
    splits : опциональный subset HF-сплитов. None = все 4.
    limit_examples_per_split : максимум HF-строк (context'ов) на split.
    limit_questions_per_example : максимум вопросов из каждой строки.
    chunk_threshold_chars : длиннее этого — порежем на chunks. None = выключено.
    chunk_max_chars : размер одного chunk'а в символах.
    """
    from datasets import load_dataset

    splits_to_load = splits if splits is not None else list(_HF_SPLITS.keys())

    tasks: list[dict[str, Any]] = []
    for hf_split in splits_to_load:
        if hf_split not in _HF_SPLITS:
            raise ValueError(
                f"Unknown HF split {hf_split!r}; expected one of {list(_HF_SPLITS)}"
            )
        category = _HF_SPLITS[hf_split]
        metric = _CATEGORY_METRICS[category]

        ds = load_dataset(HF_DATASET_ID, split=hf_split)
        if limit_examples_per_split is not None:
            ds = ds.select(range(min(limit_examples_per_split, len(ds))))

        for ex_idx, row in enumerate(ds):
            context = str(row["context"])
            questions = row.get("questions") or []
            answers = row.get("answers") or []

            # Multi-turn chunking: если длинный context — режем и сохраняем chunks.
            chunks: list[str] = []
            stored_context = context
            if (
                chunk_threshold_chars is not None
                and len(context) > chunk_threshold_chars
            ):
                from harnes.eval.multi_turn import chunk_text

                chunks = chunk_text(context, max_chars=chunk_max_chars)
                # В description Goal'а context уже не пихаем — слишком длинно.
                stored_context = ""

            n_questions = (
                min(len(questions), limit_questions_per_example)
                if limit_questions_per_example is not None
                else len(questions)
            )
            for qi in range(n_questions):
                if qi >= len(answers):
                    break
                question = str(questions[qi])
                accepted = answers[qi] if isinstance(answers[qi], list) else [answers[qi]]
                accepted_strs = [str(a) for a in accepted if a]
                if not accepted_strs:
                    continue
                task_row: dict[str, Any] = {
                    "task_id": f"{category}_{ex_idx:03d}_q{qi:03d}",
                    "category": category,
                    "context": stored_context,
                    "question": question,
                    "expected_answer": accepted_strs if len(accepted_strs) > 1 else accepted_strs[0],
                    "metric": metric,
                }
                if chunks:
                    task_row["chunks"] = chunks
                tasks.append(task_row)

    return tasks

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
            chunks = t.get("chunks") or []  # multi-turn injection
            question = str(t["question"])
            expected_raw = t["expected_answer"]
            expected_display = (
                "/".join(map(str, expected_raw))
                if isinstance(expected_raw, list)
                else str(expected_raw)
            )

            # Goal.description: для multi-turn вариант — только вопрос +
            # инструкция использовать recall_memory. Для short-context — старый
            # вариант с inline-контекстом.
            if chunks:
                description = (
                    f"Memory task [{t.get('category', '?')}]. "
                    f"Your task-scoped memory has been pre-loaded with {len(chunks)} "
                    "chunks of relevant context. Use the tool_id=recall_memory "
                    'with args={"query": "<short query>", "k": 5} to search them.\n\n'
                    f"Question: {question}\n\n"
                    "Respond by calling tool_id=finish with "
                    'args={"final_state": {"answer": "<your concise answer>"}}.'
                )
            else:
                description = (
                    f"Memory task [{t.get('category', '?')}]. "
                    f"Context (remember this):\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Respond by calling tool_id=finish with "
                    'args={"final_state": {"answer": "<your concise answer>"}}.'
                )

            goal_metadata: dict[str, Any] = {
                "benchmark": self.name,
                "task_id": task_id,
                "expected_answer": expected_raw,
                "category": t.get("category", ""),
                "metric": self.metric,
            }
            if chunks:
                goal_metadata["chunks"] = chunks

            goal = Goal(
                description=description,
                goal_class=GoalClass.TASK,
                predicate_of_success=JudgePredicate(
                    criterion=(
                        f"trajectory.final_state.answer contains the expected value "
                        f"{expected_display!r} (case-insensitive substring)"
                    )
                ),
                origin=Origin.OPERATOR,
                originator=f"benchmark:{self.name}",
                metadata=goal_metadata,
            )
            yield task_id, goal

    def verify(self, task_id: str, trajectory: Trajectory) -> tuple[bool, str | None]:
        """Сверяет trajectory.final_state с expected_answer по выбранной метрике.

        expected_answer может быть строкой ИЛИ списком (multi-valid match —
        достаточно совпасть с одним из вариантов). HF MAB чаще даёт список.
        """
        task = next((t for t in self._tasks if str(t["task_id"]) == task_id), None)
        if task is None:
            return False, "unknown_task"

        raw_expected = task["expected_answer"]
        accepted = (
            [str(x) for x in raw_expected]
            if isinstance(raw_expected, list)
            else [str(raw_expected)]
        )
        metric = task.get("metric", self.metric)

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
            ok = any(e.lower() in answer.lower() for e in accepted)
            return ok, None if ok else "substring_not_found"
        if metric == "exact_match":
            normalized = answer.strip().lower()
            ok = any(normalized == e.strip().lower() for e in accepted)
            return ok, None if ok else "exact_mismatch"

        log.warning("memory_agent_bench.unknown_metric", metric=metric)
        return False, f"unknown_metric:{metric}"
