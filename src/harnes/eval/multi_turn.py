"""Multi-turn chunk-injection для MemoryAgentBench-style задач.

См. v0.3 #27.

Проблема: контексты в MAB бывают 100k+ токенов — single-turn инжекция
(всё в Goal.description) даёт BadRequestError на ll-router (gemma-26b-a4b,
ctx 64k или даже 128k не помогает).

Решение: «inject once, query multiple times» — длинный context режется на
chunks (~2000 символов), каждый кладётся в **task-scoped in-memory store**
с эмбеддингом. Агенту даётся специальный tool `recall_memory(query, k)`,
который ищет top-k релевантных chunks. Goal.description содержит только
вопрос + инструкцию использовать recall.

Изоляция per-task — fresh InMemoryChunkStore + fresh ToolRegistry. Никакой
утечки между задачами.
"""
from __future__ import annotations

import math
from typing import Any

import structlog
from pydantic import BaseModel, Field

from harnes.tools.registry import ToolRegistry
from harnes.tools.schema import BaseIrreversibility, RetryPolicy, Tool, ToolCategory

log = structlog.get_logger()


# ---------- Chunker ----------


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 100) -> list[str]:
    """Разбивает длинный текст на overlapping chunks с учётом естественных
    границ (\\n\\n → \\n → '. ' → ' ').
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # Ищем разумную границу в последних ~30% окна.
            window_start = start + int(max_chars * 0.7)
            best = -1
            for sep in ("\n\n", "\n", ". ", " "):
                idx = text.rfind(sep, window_start, end)
                if idx > best:
                    best = idx + len(sep)
                    if sep in ("\n\n", "\n"):
                        break  # сильная граница — берём сразу
            if best > start:
                end = best
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# ---------- Task-scoped in-memory store ----------


class InMemoryChunkStore:
    """Простое in-memory хранилище chunk'ов с similarity-поиском.

    Эмбеддинги считаются через harnes.llm.embeddings.embed (fastembed
    multilingual mpnet). Cosine similarity для retrieval. Не персистентно —
    создаётся per-task, выкидывается после.
    """

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []

    def add_chunks(self, chunks: list[str]) -> None:
        if not chunks:
            return
        from harnes.llm.embeddings import embed

        vecs = embed(chunks)
        self.chunks.extend(chunks)
        self.embeddings.extend(vecs)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not self.chunks:
            return []
        from harnes.llm.embeddings import embed

        qv = embed([query])[0]
        scored = [(c, _cosine(qv, e)) for c, e in zip(self.chunks, self.embeddings)]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# ---------- recall_memory tool factory ----------


class _RecallArgs(BaseModel):
    query: str = Field(description="Natural-language query to search the memory store.")
    k: int = Field(default=5, ge=1, le=20, description="Number of top matches.")


class _RecallResult(BaseModel):
    hits: list[dict[str, Any]] = Field(
        description="List of {text, score} dicts, sorted by relevance."
    )


def make_recall_tool_for(store: InMemoryChunkStore) -> tuple[Tool, Any]:
    """Возвращает (Tool spec, impl callable), привязанный к данному store.

    Передаётся в task-scoped ToolRegistry (см. build_task_registry).
    """

    def recall_impl(args: _RecallArgs) -> _RecallResult:
        hits = store.search(args.query, k=args.k)
        return _RecallResult(
            hits=[{"text": text, "score": round(score, 4)} for text, score in hits]
        )

    tool = Tool(
        id="recall_memory",
        name="recall_memory",
        description=(
            "Search the per-task memory store (pre-populated with the task's context "
            "as chunks) for the most relevant pieces of information. Use this tool "
            "to find facts needed to answer the question."
        ),
        input_schema=_RecallArgs.model_json_schema(),
        output_schema=_RecallResult.model_json_schema(),
        base_irreversibility=BaseIrreversibility.NEVER,
        side_effects="None — reads only.",
        category=ToolCategory.INFO,
        retry_policy=RetryPolicy(),
        timeout_seconds=15.0,
        implementation_ref="harnes.eval.multi_turn.recall_impl_bound",
    )
    return tool, recall_impl


def build_task_registry(
    chunks: list[str],
    base_tool_ids: list[str] | None = None,
) -> tuple[ToolRegistry, InMemoryChunkStore]:
    """Создаёт task-scoped ToolRegistry с:
    - read_file и write_file (опционально, для backward-compat skill)
    - recall_memory (свежий InMemoryChunkStore с пред-заполненными chunks)

    Возвращает (registry, store) — store даётся для тестирования; в обычном
    флоу нужен только registry.
    """
    from harnes.tools.builtin.io import (
        READ_FILE_TOOL,
        ReadFileArgs,
        ReadFileResult,
        WRITE_FILE_TOOL,
        WriteFileArgs,
        WriteFileResult,
        read_file_impl,
        write_file_impl,
    )

    registry = ToolRegistry()
    base = base_tool_ids if base_tool_ids is not None else ["read_file", "write_file"]
    if "read_file" in base:
        registry.register(READ_FILE_TOOL, read_file_impl, ReadFileArgs, ReadFileResult)
    if "write_file" in base:
        registry.register(
            WRITE_FILE_TOOL, write_file_impl, WriteFileArgs, WriteFileResult
        )

    store = InMemoryChunkStore()
    store.add_chunks(chunks)
    tool, impl = make_recall_tool_for(store)
    registry.register(tool, impl, _RecallArgs, _RecallResult)

    log.debug(
        "multi_turn.task_registry.built",
        chunks=len(chunks),
        tools=registry.list_ids(),
    )
    return registry, store
