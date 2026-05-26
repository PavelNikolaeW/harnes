"""Tests for harnes.memory.world (Graphiti+Neo4j wrapper).

Unit-тесты мокают graphiti_core полностью — реальный Neo4j не нужен.
Real integration test — в tests/test_e2e_smoke.py (отдельный смоук с пометкой
о доступности Neo4j).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harnes.memory.world import WorldModelStore


@pytest.fixture
def world() -> WorldModelStore:
    return WorldModelStore("bolt://localhost:7687", "neo4j", "test")


# ---------- Construction ----------


def test_world_construction_does_not_connect() -> None:
    """Конструктор не должен сразу лезть в Neo4j — lazy init."""
    w = WorldModelStore("bolt://nowhere:9999", "x", "y")
    assert w._graphiti is None
    assert w._initialised is False


# ---------- add_episode ----------


def test_add_episode_calls_graphiti(world: WorldModelStore) -> None:
    fake = MagicMock()
    fake.add_episode = AsyncMock()
    fake.build_indices_and_constraints = AsyncMock()
    world._graphiti = fake
    world._initialised = True  # skip init

    world.add_episode(name="ep1", episode_body="Goal X completed successfully")

    fake.add_episode.assert_called_once()
    kwargs = fake.add_episode.call_args.kwargs
    assert kwargs["name"] == "ep1"
    assert kwargs["episode_body"] == "Goal X completed successfully"
    assert kwargs["source_description"] == "harnes"


def test_add_episode_swallows_exceptions(world: WorldModelStore) -> None:
    """add_episode не должен бросать — world_update это side-channel."""
    fake = MagicMock()
    fake.add_episode = AsyncMock(side_effect=RuntimeError("neo4j down"))
    world._graphiti = fake
    world._initialised = True

    # Should not raise
    world.add_episode(name="ep1", episode_body="x")


# ---------- search ----------


def test_search_converts_graphiti_results(world: WorldModelStore) -> None:
    """Результаты Graphiti конвертируются в унифицированный dict-format."""
    # Минимальный mock-объект, похожий на Graphiti EntityNode/Edge.
    fake_node = MagicMock()
    fake_node.uuid = "abc-123"
    fake_node.name = "Service X"
    fake_node.summary = "Running smoothly"

    fake = MagicMock()
    fake.search = AsyncMock(return_value=[fake_node])
    fake.build_indices_and_constraints = AsyncMock()
    world._graphiti = fake
    world._initialised = True

    results = world.search("status of Service X")

    assert len(results) == 1
    r = results[0]
    assert r["id"] == "abc-123"
    assert "name" in r["properties"]
    assert r["properties"]["name"] == "Service X"


def test_search_empty_on_exception(world: WorldModelStore) -> None:
    fake = MagicMock()
    fake.search = AsyncMock(side_effect=RuntimeError("neo4j unreachable"))
    fake.build_indices_and_constraints = AsyncMock()
    world._graphiti = fake
    world._initialised = True

    results = world.search("anything")
    assert results == []


# ---------- close ----------


def test_close_calls_graphiti_close(world: WorldModelStore) -> None:
    fake = MagicMock()
    fake.close = AsyncMock()
    world._graphiti = fake

    world.close()
    fake.close.assert_called_once()


def test_close_when_not_initialised_is_safe(world: WorldModelStore) -> None:
    # Не должно крашить если Graphiti ещё не создан.
    world.close()
