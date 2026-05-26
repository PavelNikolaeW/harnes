"""Tests for v1.0 #34: recall_memory builtin tool wired to MemoryRouter."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from harnes.memory.episodic import EpisodicStore
from harnes.memory.router import MemoryRouter
from harnes.memory.schema import EpisodicRecord, MemoryBundle, SemanticRecord, WorldNode
from harnes.tools.builtin.recall import (
    RecallArgs,
    RecallResult,
    build_runtime_registry,
    make_recall_tool_for_router,
)


# ---------- Schemas ----------


def test_recall_args_validation() -> None:
    args = RecallArgs(query="x")
    assert args.k == 5
    assert args.types is None


def test_recall_args_k_bounds() -> None:
    with pytest.raises(Exception):
        RecallArgs(query="x", k=0)
    with pytest.raises(Exception):
        RecallArgs(query="x", k=21)


# ---------- Tool factory ----------


def test_make_recall_tool_returns_spec_and_impl() -> None:
    router = MagicMock(spec=MemoryRouter)
    tool, impl = make_recall_tool_for_router(router)
    assert tool.id == "recall_memory"
    assert tool.category.value == "info"
    assert tool.implementation_ref.startswith("harnes.tools.builtin.recall.")
    # impl callable
    assert callable(impl)


def test_recall_impl_returns_empty_when_router_returns_empty() -> None:
    router = MagicMock(spec=MemoryRouter)
    router.recall.return_value = MemoryBundle()
    _, impl = make_recall_tool_for_router(router)

    result = impl(RecallArgs(query="any"))
    assert isinstance(result, RecallResult)
    assert result.episodic == []
    assert result.semantic == []
    assert result.world == []


def test_recall_impl_maps_episodic_records() -> None:
    router = MagicMock(spec=MemoryRouter)
    rec = EpisodicRecord(
        trajectory_id=uuid4(),
        step_id=uuid4(),
        goal_id=uuid4(),
        timestamp=datetime.now(UTC),
        step_type="thought",
        content={"text": "Recalled this earlier observation"},
    )
    router.recall.return_value = MemoryBundle(episodic=[rec])
    _, impl = make_recall_tool_for_router(router)

    result = impl(RecallArgs(query="observation"))
    assert len(result.episodic) == 1
    hit = result.episodic[0]
    assert "earlier observation" in hit.text or "Recalled" in hit.text
    assert hit.metadata["step_type"] == "thought"
    assert hit.metadata["trajectory_id"] == str(rec.trajectory_id)


def test_recall_impl_maps_semantic_and_world() -> None:
    router = MagicMock(spec=MemoryRouter)
    router.recall.return_value = MemoryBundle(
        semantic=[
            SemanticRecord(
                id="s1",
                text="A fact about Atlantis",
                embedding=None,
                metadata={"source": "wiki"},
            ),
        ],
        world=[
            WorldNode(
                id="n1",
                labels=["Person"],
                properties={"name": "Alice", "summary": "owns Atlantis"},
            ),
        ],
    )
    _, impl = make_recall_tool_for_router(router)
    result = impl(RecallArgs(query="atlantis"))
    assert len(result.semantic) == 1
    assert "Atlantis" in result.semantic[0].text
    assert result.semantic[0].metadata["source"] == "wiki"

    assert len(result.world) == 1
    w = result.world[0]
    assert "Person" in w.text
    assert "Alice" in w.text


def test_recall_impl_handles_router_exception_gracefully() -> None:
    """Если router бросает — возвращается пустой RecallResult, не raise."""
    router = MagicMock(spec=MemoryRouter)
    router.recall.side_effect = RuntimeError("backend down")
    _, impl = make_recall_tool_for_router(router)

    result = impl(RecallArgs(query="x"))
    assert isinstance(result, RecallResult)
    assert result.episodic == []
    assert result.semantic == []
    assert result.world == []


def test_recall_impl_passes_types_filter() -> None:
    """args.types фильтрует, какие backends опрашиваются."""
    router = MagicMock(spec=MemoryRouter)
    router.recall.return_value = MemoryBundle()
    _, impl = make_recall_tool_for_router(router)

    impl(RecallArgs(query="x", types=["episodic", "semantic"]))
    # router.recall был вызван с types=[MemoryType.EPISODIC, MemoryType.SEMANTIC]
    call_kwargs = router.recall.call_args.kwargs
    from harnes.memory.schema import MemoryType

    assert call_kwargs["types"] == [MemoryType.EPISODIC, MemoryType.SEMANTIC]


def test_recall_impl_ignores_unknown_types() -> None:
    """Неизвестные типы отбрасываются."""
    router = MagicMock(spec=MemoryRouter)
    router.recall.return_value = MemoryBundle()
    _, impl = make_recall_tool_for_router(router)

    impl(RecallArgs(query="x", types=["unknown", "made_up"]))
    # All filtered out → types=None
    call_kwargs = router.recall.call_args.kwargs
    assert call_kwargs["types"] is None


# ---------- build_runtime_registry ----------


def test_runtime_registry_has_io_tools_only_when_no_router() -> None:
    registry = build_runtime_registry(router=None)
    ids = set(registry.list_ids())
    assert "read_file" in ids
    assert "write_file" in ids
    assert "recall_memory" not in ids


def test_runtime_registry_has_recall_when_router_provided() -> None:
    router = MagicMock(spec=MemoryRouter)
    registry = build_runtime_registry(router=router)
    ids = set(registry.list_ids())
    assert "read_file" in ids
    assert "write_file" in ids
    assert "recall_memory" in ids


def test_runtime_registry_recall_only_without_io() -> None:
    router = MagicMock(spec=MemoryRouter)
    registry = build_runtime_registry(router=router, include_io=False)
    ids = set(registry.list_ids())
    assert ids == {"recall_memory"}


# ---------- Integration: tool invocation через registry ----------


def test_registry_invoke_recall_memory_returns_observation(tmp_path: Path) -> None:
    """End-to-end: registry.invoke('recall_memory', ...) → ObservationStep с SUCCESS."""
    # Реальный episodic с одной записью.
    episodic = EpisodicStore(tmp_path / "ep")
    router = MemoryRouter(episodic=episodic)

    registry = build_runtime_registry(router=router)
    obs = registry.invoke("recall_memory", {"query": "anything", "k": 3})

    # Outcome SUCCESS даже если episodic пустой — просто пустой результат.
    assert obs.outcome.value == "success"
    assert obs.payload is not None
    assert "episodic" in obs.payload
    assert "semantic" in obs.payload
    assert "world" in obs.payload


def test_registry_invoke_recall_with_invalid_args_schema_error() -> None:
    """k вне диапазона → SCHEMA_ERROR."""
    router = MagicMock(spec=MemoryRouter)
    registry = build_runtime_registry(router=router)
    obs = registry.invoke("recall_memory", {"query": "x", "k": 999})
    assert obs.outcome.value == "schema_error"


# ---------- Skill bundle update sanity ----------


def test_general_skill_allows_recall_memory() -> None:
    """general.yaml должен включать recall_memory в allowed_tools после v1.0 #34."""
    import yaml
    skill_path = Path(__file__).resolve().parent.parent / "skills" / "general.yaml"
    with skill_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "recall_memory" in data["allowed_tools"]
