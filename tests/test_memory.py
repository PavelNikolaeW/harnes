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


def _make_trajectory_with_thought(text: str) -> Trajectory:
    return Trajectory(
        goal_id=uuid4(),
        steps=[ThoughtStep(text=text, cost=Cost(tokens=10))],
        status=TrajectoryStatus.SUCCESS,
        final_state={"text": text},
        total_cost=Cost(tokens=10),
        ended_at=datetime.now(UTC),
    )


def test_episodic_search_by_terms_finds_match_over_recency(
    episodic: EpisodicStore,
) -> None:
    """Шаг с релевантным content должен подниматься выше свежих шумовых шагов."""
    target = _make_trajectory_with_thought("I multiplied 7 by 8 to get 56")
    episodic.write_trajectory(target)
    for _ in range(10):
        episodic.write_trajectory(_make_trajectory())  # noise: "read the file"

    hits = episodic.search_steps_by_terms(terms=["multiplied"], limit=5)
    assert hits, "search returned no hits despite a matching step"
    assert any(r["trajectory_id"] == str(target.id) for r in hits)


def test_episodic_search_by_terms_empty_terms_returns_empty(
    episodic: EpisodicStore,
) -> None:
    episodic.write_trajectory(_make_trajectory())
    assert episodic.search_steps_by_terms(terms=[], limit=5) == []


def test_episodic_search_scores_by_hit_count(episodic: EpisodicStore) -> None:
    """Шаг с двумя term-hits должен ранжироваться выше шага с одним."""
    one_hit = _make_trajectory_with_thought("just multiplied two numbers")
    two_hit = _make_trajectory_with_thought("computed and multiplied — the result was 56")
    episodic.write_trajectory(one_hit)
    episodic.write_trajectory(two_hit)

    hits = episodic.search_steps_by_terms(terms=["multiplied", "computed"], limit=5)
    assert hits[0]["trajectory_id"] == str(two_hit.id)


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


def test_router_episodic_recall_uses_query_over_recency(
    episodic: EpisodicStore,
) -> None:
    """Регрессия: prior _recall_episodic игнорировал query и возвращал последние
    N шагов. После фикса релевантная trajectory всплывает даже если её шаги
    давно вытолкнуты recency-окном."""
    target = _make_trajectory_with_thought("computed 7 multiplied by 8 = 56")
    episodic.write_trajectory(target)
    for _ in range(10):
        episodic.write_trajectory(_make_trajectory())  # noise: "read the file"

    router = MemoryRouter(episodic=episodic)
    bundle = router.recall(query="multiplied number", k=3)
    assert any(r.trajectory_id == target.id for r in bundle.episodic), (
        "router did not surface trajectory matching query — "
        "are we falling back to recency too eagerly?"
    )


def test_router_episodic_recall_falls_back_to_recency_for_stopword_query(
    episodic: EpisodicStore,
) -> None:
    """Запрос из одних стопвордов → terms=[] → fallback на recent_steps."""
    for _ in range(3):
        episodic.write_trajectory(_make_trajectory())

    router = MemoryRouter(episodic=episodic)
    bundle = router.recall(query="what is it", k=2)
    # 3 trajectories x 3 steps = 9 steps; k=2 -> 2 hits from recency
    assert len(bundle.episodic) == 2


# ---------- Vector search (BGE-M3 ANN) ----------


def test_episodic_vector_search_finds_semantic_match(
    episodic: EpisodicStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock embeddings: семантически близкие тексты → одинаковые векторы.
    Vector поиск должен возвращать шаг по тематическому матчу, не точному слову."""
    def fake_embed(texts):
        out = []
        for t in texts:
            tl = t.lower()
            if "compute" in tl or "calc" in tl or "arithmetic" in tl or "multiplied" in tl:
                v = [1.0] + [0.0] * 1023
            elif "file" in tl:
                v = [0.0, 1.0] + [0.0] * 1022
            else:
                v = [0.0] * 1024
                v[2] = 1.0
            out.append(v)
        return out

    from harnes.llm import embeddings as embeddings_mod
    monkeypatch.setattr(embeddings_mod, "embed", fake_embed)

    target = _make_trajectory_with_thought("computed 7 multiplied by 8 = 56")
    episodic.write_trajectory(target)
    for _ in range(5):
        noise = _make_trajectory_with_thought("read file /tmp/x.txt")
        episodic.write_trajectory(noise)

    hits = episodic.search_steps_by_vector(query="arithmetic operation result", limit=3)
    assert hits, "vector search should find semantically-similar step"
    assert hits[0]["trajectory_id"] == str(target.id)


def test_episodic_vector_search_returns_empty_when_no_embeddings(
    episodic: EpisodicStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Если embeddings table пустая (никто не писал) — vector search возвращает []."""
    def fake_embed(texts):
        return [[0.5] * 1024 for _ in texts]

    from harnes.llm import embeddings as embeddings_mod
    monkeypatch.setattr(embeddings_mod, "embed", fake_embed)

    assert episodic.search_steps_by_vector(query="anything", limit=5) == []


def test_episodic_vector_search_handles_dim_mismatch(
    episodic: EpisodicStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Если embed возвращает не 1024-dim — graceful return [] (не raise)."""
    def fake_embed(texts):
        return [[0.5] * 768 for _ in texts]

    from harnes.llm import embeddings as embeddings_mod
    monkeypatch.setattr(embeddings_mod, "embed", fake_embed)

    assert episodic.search_steps_by_vector(query="anything", limit=5) == []


def test_episodic_write_skips_embeddings_on_dim_mismatch(
    episodic: EpisodicStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fastembed fallback с другой dim — write не падает, embeddings table пустая."""
    def fake_embed_768(texts):
        return [[0.1] * 768 for _ in texts]

    from harnes.llm import embeddings as embeddings_mod
    monkeypatch.setattr(embeddings_mod, "embed", fake_embed_768)

    episodic.write_trajectory(_make_trajectory_with_thought("hello world"))

    # Меняем mock на правильный dim — но в БД ничего нет с этим dim.
    monkeypatch.setattr(embeddings_mod, "embed", lambda ts: [[0.1] * 1024 for _ in ts])
    assert episodic.search_steps_by_vector(query="hello", limit=5) == []


def test_router_episodic_recall_uses_vector_first(
    episodic: EpisodicStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recall-chain: vector hit имеет приоритет над keyword/recency."""
    def fake_embed(texts):
        out = []
        for t in texts:
            v = [0.0] * 1024
            if "multiplied" in t.lower() or "arithmetic" in t.lower():
                v[0] = 1.0
            else:
                v[500] = 1.0
            out.append(v)
        return out

    from harnes.llm import embeddings as embeddings_mod
    monkeypatch.setattr(embeddings_mod, "embed", fake_embed)

    target = _make_trajectory_with_thought("multiplied 7 by 8")
    episodic.write_trajectory(target)

    router = MemoryRouter(episodic=episodic)
    # "arithmetic" семантически близок к "multiplied" — vector chain должен сработать.
    bundle = router.recall(query="arithmetic problem", k=3)
    assert any(r.trajectory_id == target.id for r in bundle.episodic)


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
