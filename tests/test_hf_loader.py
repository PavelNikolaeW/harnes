"""Tests for HF MemoryAgentBench loader.

Реальный HF-download — медленный + сетевая зависимость. Поэтому мокаем
datasets.load_dataset и тестируем нашу логику flattening.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from harnes.eval.adapters.memory_agent_bench import (
    _CATEGORY_METRICS,
    _HF_SPLITS,
    load_hf_tasks,
)


def _mock_ds(rows: list[dict]) -> MagicMock:
    """Создаёт mock-объект, похожий на HF Dataset."""
    ds = MagicMock()
    ds.__len__ = lambda self: len(rows)
    ds.__iter__ = lambda self: iter(rows)
    ds.select = lambda indices: _mock_ds([rows[i] for i in indices])
    return ds


def test_load_hf_tasks_flattens_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Каждый (context, q, answers) → отдельный task."""
    fake_data = {
        "Accurate_Retrieval": [
            {
                "context": "facts about A and B",
                "questions": ["what about A?", "what about B?", "third?"],
                "answers": [["A1"], ["B1", "B2"], ["X"]],
                "metadata": {},
            }
        ],
        "Test_Time_Learning": [
            {
                "context": "examples",
                "questions": ["q1"],
                "answers": [["ans"]],
                "metadata": {},
            },
        ],
    }

    def fake_load(_ds_id, split):
        return _mock_ds(fake_data.get(split, []))

    with patch("datasets.load_dataset", side_effect=fake_load):
        tasks = load_hf_tasks(
            splits=["Accurate_Retrieval", "Test_Time_Learning"]
        )

    # 3 q'ов из AR + 1 из TTL = 4
    assert len(tasks) == 4
    ar_tasks = [t for t in tasks if t["category"] == "accurate_retrieval"]
    assert len(ar_tasks) == 3
    assert ar_tasks[0]["task_id"].startswith("accurate_retrieval_000_q000")
    assert ar_tasks[0]["context"] == "facts about A and B"

    # Multi-valid answer (B1, B2) сохранён как list
    multi = ar_tasks[1]
    assert isinstance(multi["expected_answer"], list)
    assert multi["expected_answer"] == ["B1", "B2"]

    # Single-valid answer (A1) уплощается в строку
    single = ar_tasks[0]
    assert single["expected_answer"] == "A1"

    # Metric выбран по категории
    ttl = next(t for t in tasks if t["category"] == "test_time_learning")
    assert ttl["metric"] == "exact_match"  # из _CATEGORY_METRICS


def test_load_hf_tasks_respects_limit_questions_per_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = {
        "Conflict_Resolution": [
            {
                "context": "c",
                "questions": [f"q{i}" for i in range(20)],
                "answers": [[f"a{i}"] for i in range(20)],
                "metadata": {},
            }
        ]
    }
    with patch(
        "datasets.load_dataset",
        side_effect=lambda _id, split: _mock_ds(fake.get(split, [])),
    ):
        tasks = load_hf_tasks(
            splits=["Conflict_Resolution"],
            limit_questions_per_example=5,
        )
    assert len(tasks) == 5


def test_load_hf_tasks_respects_limit_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "Accurate_Retrieval": [
            {"context": f"c{i}", "questions": ["q"], "answers": [["a"]], "metadata": {}}
            for i in range(10)
        ]
    }
    with patch(
        "datasets.load_dataset",
        side_effect=lambda _id, split: _mock_ds(fake.get(split, [])),
    ):
        tasks = load_hf_tasks(
            splits=["Accurate_Retrieval"], limit_examples_per_split=3
        )
    assert len(tasks) == 3


def test_load_hf_tasks_unknown_split_raises() -> None:
    with pytest.raises(ValueError):
        load_hf_tasks(splits=["NoSuchSplit"])


def test_load_hf_tasks_skips_when_no_answer() -> None:
    fake = {
        "Accurate_Retrieval": [
            {
                "context": "c",
                "questions": ["q1", "q2"],
                "answers": [[], ["valid"]],
                "metadata": {},
            }
        ]
    }
    with patch(
        "datasets.load_dataset",
        side_effect=lambda _id, split: _mock_ds(fake.get(split, [])),
    ):
        tasks = load_hf_tasks(splits=["Accurate_Retrieval"])
    # Первый вопрос (без ответа) пропущен, второй взят
    assert len(tasks) == 1
    assert tasks[0]["question"] == "q2"


def test_category_metrics_mapping() -> None:
    """Проверка соответствия README MAB: AR/CR → substring, TTL/LRU → exact."""
    assert _CATEGORY_METRICS["accurate_retrieval"] == "substring_exact_match"
    assert _CATEGORY_METRICS["conflict_resolution"] == "substring_exact_match"
    assert _CATEGORY_METRICS["test_time_learning"] == "exact_match"
    assert _CATEGORY_METRICS["long_range_understanding"] == "exact_match"


# ---------- Verify with multi-valid answers ----------


def test_adapter_verify_accepts_one_of_multi_valid() -> None:
    """Если expected_answer — список, match с любым вариантом считается ok."""
    from uuid import uuid4

    from harnes.eval import MemoryAgentBenchAdapter
    from harnes.react.schema import Cost, Trajectory, TrajectoryStatus

    a = MemoryAgentBenchAdapter(
        tasks=[
            {
                "task_id": "x",
                "context": "",
                "question": "?",
                "expected_answer": ["alpha", "beta", "gamma"],
                "metric": "substring_exact_match",
            }
        ]
    )

    traj = Trajectory(
        goal_id=uuid4(),
        status=TrajectoryStatus.SUCCESS,
        final_state={"answer": "the answer is beta indeed"},
        total_cost=Cost(),
    )
    ok, _ = a.verify("x", traj)
    assert ok is True
