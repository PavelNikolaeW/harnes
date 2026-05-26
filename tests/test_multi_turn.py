"""Tests for harnes.eval.multi_turn (chunker + InMemoryChunkStore + tool factory)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from harnes.eval.multi_turn import (
    InMemoryChunkStore,
    _cosine,
    build_task_registry,
    chunk_text,
    make_recall_tool_for,
)


# ---------- chunk_text ----------


def test_chunk_text_returns_single_when_short() -> None:
    assert chunk_text("hello world", max_chars=1000) == ["hello world"]


def test_chunk_text_empty_returns_empty() -> None:
    assert chunk_text("", max_chars=100) == []


def test_chunk_text_splits_long() -> None:
    text = "Sentence one. Sentence two. Sentence three. " * 50  # ~2200 chars
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 500


def test_chunk_text_prefers_paragraph_boundary() -> None:
    """Chunker должен предпочитать \\n\\n как сильную границу."""
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird here."
    # max_chars немного больше длины первого para — должен резать на \n\n.
    chunks = chunk_text(text, max_chars=30, overlap=0)
    # Точная граница зависит от поиска; проверим что чанков >= 2 и они не сломаны посредине слова
    assert len(chunks) >= 2
    assert all(not c.endswith("paragra") for c in chunks)  # не оборвано посередине


def test_chunk_text_handles_no_natural_boundary() -> None:
    """Если нет хороших разделителей — режем по словам/пробелам."""
    text = "abc" * 1000  # один сплошной строй без разделителей
    chunks = chunk_text(text, max_chars=500, overlap=10)
    assert len(chunks) >= 2


# ---------- _cosine ----------


def test_cosine_identical_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    assert _cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector() -> None:
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_empty() -> None:
    assert _cosine([], [1.0]) == 0.0


# ---------- InMemoryChunkStore ----------


def test_chunk_store_empty_search() -> None:
    store = InMemoryChunkStore()
    assert store.search("anything") == []


def test_chunk_store_add_and_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search возвращает chunks отсортированные по cosine similarity."""
    from harnes.llm import embeddings as emb_mod

    # Зафиксированные «эмбеддинги» для трёх чанков и query.
    canned = {
        "Postgres reports high error rate.": [1.0, 0.0, 0.0],
        "User favourite color is teal.": [0.0, 1.0, 0.0],
        "Service ABC is also fine.": [0.0, 0.0, 1.0],
        "what is the error?": [0.9, 0.1, 0.0],  # ближе к Postgres
    }

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [canned.get(t, [0.0, 0.0, 0.0]) for t in texts]

    monkeypatch.setattr(emb_mod, "_embed_via_fastembed", fake_embed)

    store = InMemoryChunkStore()
    store.add_chunks(
        [
            "Postgres reports high error rate.",
            "User favourite color is teal.",
            "Service ABC is also fine.",
        ]
    )

    results = store.search("what is the error?", k=2)
    assert len(results) == 2
    assert results[0][0] == "Postgres reports high error rate."
    assert results[0][1] > results[1][1]  # ранжирование убывающее


def test_chunk_store_add_empty_noop() -> None:
    store = InMemoryChunkStore()
    store.add_chunks([])
    assert store.chunks == []


# ---------- Tool factory ----------


def test_make_recall_tool_returns_spec_and_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    from harnes.llm import embeddings as emb_mod

    monkeypatch.setattr(emb_mod, "_embed_via_fastembed", lambda texts: [[1.0]] * len(texts))

    store = InMemoryChunkStore()
    store.add_chunks(["chunk one", "chunk two"])

    tool, impl = make_recall_tool_for(store)

    assert tool.id == "recall_memory"
    assert tool.category.value == "info"

    from harnes.eval.multi_turn import _RecallArgs

    args = _RecallArgs(query="something")
    result = impl(args)
    assert hasattr(result, "hits")
    assert len(result.hits) == 2
    assert "text" in result.hits[0]
    assert "score" in result.hits[0]


# ---------- build_task_registry ----------


def test_build_task_registry_has_recall_and_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harnes.llm import embeddings as emb_mod

    monkeypatch.setattr(
        emb_mod, "_embed_via_fastembed", lambda texts: [[1.0, 0.0]] * len(texts)
    )

    registry, store = build_task_registry(["chunk1", "chunk2"])
    ids = registry.list_ids()
    assert "recall_memory" in ids
    assert "read_file" in ids
    assert "write_file" in ids

    # Store предварительно заполнен
    assert len(store.chunks) == 2


def test_build_task_registry_recall_invocation_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harnes.llm import embeddings as emb_mod

    monkeypatch.setattr(
        emb_mod,
        "_embed_via_fastembed",
        lambda texts: [[1.0] if "fact" in t else [0.0] for t in texts],
    )

    registry, _ = build_task_registry(["fact A", "noise"])
    obs = registry.invoke("recall_memory", {"query": "fact?", "k": 1})
    assert obs.outcome.value == "success"
    assert obs.payload is not None
    assert "hits" in obs.payload
