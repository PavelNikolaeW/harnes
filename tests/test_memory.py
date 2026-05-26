"""Tests for memory layer.

EpisodicStore (LanceDB) тестируется реально на tmp_path.
SemanticStore (Qdrant) и WorldModelStore (Graphiti) — мокаются.
MemoryRouter — с in-memory episodic + моки остальных.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from harnes.memory.episodic import EpisodicStore
from harnes.memory.router import MemoryRouter
from harnes.memory.schema import MemoryType
from harnes.react.schema import (
    ActionStep,
    Cost,
    ObservationOutcome,
    ObservationStep,
    ThoughtStep,
    Trajectory,
    TrajectoryStatus,
)


# ============================================================
# EpisodicStore (LanceDB, real, embedded)
# ============================================================


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "lancedb")


def _make_trajectory() -> Trajectory:
    goal_id = uuid4()
    return Trajectory(
        goal_id=goal_id,
        steps=[
            ThoughtStep(text="I should read the file", cost=Cost(tokens=10)),
            ActionStep(
                tool_id="read_file", args={"path": "/tmp/x"}, cost=Cost(tokens=15)
            ),
            ObservationStep(
                outcome=ObservationOutcome.SUCCESS,
                payload={"content": "hello"},
                cost=Cost(tokens=5),
            ),
        ],
        status=TrajectoryStatus.SUCCESS,
        final_state={"content": "hello"},
        total_cost=Cost(tokens=30, latency_seconds=1.2),
        ended_at=datetime.now(UTC),
    )


def test_episodic_write_and_read_trajectory(episodic: EpisodicStore) -> None:
    traj = _make_trajectory()
    episodic.write_trajectory(traj)

    meta = episodic.get_trajectory_meta(traj.id)
    assert meta is not None
    assert meta["id"] == str(traj.id)
    assert meta["status"] == "success"
    assert meta["total_cost_tokens"] == 30

    steps = episodic.get_steps(traj.id)
    assert len(steps) == 3
    types = [s["step_type"] for s in steps]
    assert types == ["thought", "action", "observation"]


def test_episodic_list_by_goal(episodic: EpisodicStore) -> None:
    traj1 = _make_trajectory()
    traj2 = _make_trajectory()
    traj2.goal_id = traj1.goal_id  # same goal

    episodic.write_trajectory(traj1)
    episodic.write_trajectory(traj2)

    items = episodic.list_trajectories_for_goal(traj1.goal_id)
    assert len(items) == 2


def test_episodic_recent_steps(episodic: EpisodicStore) -> None:
    for _ in range(3):
        episodic.write_trajectory(_make_trajectory())

    recent = episodic.recent_steps(limit=5)
    assert len(recent) == 5  # we have 9 total, asked for 5
    # сортировка по timestamp desc
    timestamps = [r["timestamp"] for r in recent]
    assert timestamps == sorted(timestamps, reverse=True)


def test_episodic_persistence_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "persist"
    e1 = EpisodicStore(db_path)
    traj = _make_trajectory()
    e1.write_trajectory(traj)

    e2 = EpisodicStore(db_path)
    meta = e2.get_trajectory_meta(traj.id)
    assert meta is not None


# ============================================================
# MemoryRouter (с моками для semantic/world)
# ============================================================


def test_router_recall_episodic_only(episodic: EpisodicStore) -> None:
    """С только-episodic бэкендом — bundle содержит только episodic."""
    traj = _make_trajectory()
    episodic.write_trajectory(traj)

    router = MemoryRouter(episodic=episodic)
    bundle = router.recall(query="hello", types=[MemoryType.EPISODIC], k=5)
    assert len(bundle.episodic) > 0
    assert bundle.semantic == []
    assert bundle.world == []


def test_router_skips_missing_backends() -> None:
    """Если бэкенд None — соответствующий тип просто не возвращается."""
    router = MemoryRouter()  # все None
    bundle = router.recall(query="x")
    assert bundle.episodic == []
    assert bundle.semantic == []
    assert bundle.world == []


def test_router_calls_semantic_with_embedding(
    episodic: EpisodicStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SemanticStore.search получает эмбеддинг от harnes.llm.embeddings.embed."""
    mock_semantic = MagicMock()
    mock_semantic.search.return_value = [
        {"id": "x1", "score": 0.9, "payload": {"text": "fact A"}},
    ]

    # Замокать embed чтобы не дёргать fastembed
    from harnes.llm import embeddings as emb_mod

    monkeypatch.setattr(emb_mod, "_embed_via_fastembed", lambda texts: [[0.1] * 1024])

    router = MemoryRouter(episodic=episodic, semantic=mock_semantic)
    bundle = router.recall(query="what is X", types=[MemoryType.SEMANTIC], k=3)

    assert mock_semantic.search.called
    call_kwargs = mock_semantic.search.call_args.kwargs
    # Vector передан длиной 1024
    args_passed = mock_semantic.search.call_args
    # query_vector передан в args[0] или kwargs
    if args_passed.args:
        assert len(args_passed.args[0]) == 1024
    else:
        assert len(call_kwargs["query_vector"]) == 1024

    assert len(bundle.semantic) == 1
    assert bundle.semantic[0].id == "x1"
    assert bundle.semantic[0].text == "fact A"


def test_router_world_stub_returns_empty(episodic: EpisodicStore) -> None:
    """WorldModelStore — stub в v0, всегда возвращает []."""
    from harnes.memory.world import WorldModelStore

    world = WorldModelStore("bolt://nowhere:7687", "u", "p")
    router = MemoryRouter(episodic=episodic, world=world)
    bundle = router.recall(query="x", types=[MemoryType.WORLD])
    assert bundle.world == []


# ============================================================
# SemanticStore — мокаем Qdrant client
# ============================================================


def test_semantic_upsert_calls_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    from harnes.memory import semantic as sem_mod

    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    def fake_qdrant(**kwargs):
        return mock_client

    monkeypatch.setattr(sem_mod, "QdrantClient", fake_qdrant)

    store = sem_mod.SemanticStore(url="http://fake:6333")
    store.upsert([{"id": "a", "vector": [0.1, 0.2], "payload": {"text": "hi"}}])

    assert mock_client.upsert.called
