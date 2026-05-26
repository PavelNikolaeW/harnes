"""Concrete BenchmarkAdapter implementations.

См. harnes.eval.harness.BenchmarkAdapter Protocol.
"""
from harnes.eval.adapters.memory_agent_bench import (
    MemoryAgentBenchAdapter,
    load_hf_tasks,
)

__all__ = ["MemoryAgentBenchAdapter", "load_hf_tasks"]
