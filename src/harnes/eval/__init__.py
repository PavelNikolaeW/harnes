"""Evaluation harness — интеграция с MemoryAgentBench (ICLR 2026).

См. `agent_architecture.html` § 17.

В v0 — скелет: типы для результата + adapter-интерфейс. Реальная привязка к
бенчмарку (https://github.com/HUST-AI-HYZ/MemoryAgentBench) включается через
external task #14 / отдельную итерацию.
"""
from harnes.eval.harness import (
    BenchmarkAdapter,
    EvalResult,
    PerTaskResult,
    run_evaluation,
)

__all__ = ["BenchmarkAdapter", "EvalResult", "PerTaskResult", "run_evaluation"]
