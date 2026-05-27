"""Pytest fixtures для webui smoke + behaviour tests.

Сценарий изоляции:
- tmp_data_dir(tmp_path) — `data/` subdir под pytest tmp_path.
- webui_env(monkeypatch, tmp_data_dir) — выставляет все ENV-переменные
  (`GOAL_STORE__SQLITE_PATH`, `METACYCLE__JOURNAL_DB_PATH`, ...),
  reset'ит singleton-ы `harnes.config._settings` и
  `harnes.webui.config._settings` чтобы фабрики перечитали env.
- webui_app(webui_env) — `create_app()` (lifespan ещё не дёрнут).
- client(webui_app) — `with TestClient(app) as c` — `with` важен,
  он запускает lifespan и инициализирует stores в app.state.
- goal_repo / pending_goal / pending_approval_goal — данные для goal-тестов.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
)
from harnes.goals.store import GoalRepository


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """`data/` subdir под pytest tmp_path. Каталог создан, пустой."""
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def webui_env(monkeypatch: pytest.MonkeyPatch, tmp_data_dir: Path) -> Path:
    """Все store-пути → tmp_data_dir. Singletons settings сбрасываются."""
    # Файловые stores (SQLite + LanceDB).
    monkeypatch.setenv("GOAL_STORE__SQLITE_PATH", str(tmp_data_dir / "goals.db"))
    monkeypatch.setenv(
        "METACYCLE__JOURNAL_DB_PATH", str(tmp_data_dir / "metacycle_journal.db")
    )
    monkeypatch.setenv(
        "METACYCLE__COMMANDS_DB_PATH", str(tmp_data_dir / "web_commands.db")
    )
    monkeypatch.setenv("MEMORY__LANCEDB_PATH", str(tmp_data_dir / "lancedb"))
    monkeypatch.setenv("EVAL__HISTORY_DB_PATH", str(tmp_data_dir / "eval_history.db"))
    monkeypatch.setenv(
        "PROCEDURAL_STORE__SQLITE_PATH", str(tmp_data_dir / "skill_metrics.db")
    )
    bundles = tmp_data_dir / "skills"
    bundles.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PROCEDURAL_STORE__BUNDLES_DIR", str(bundles))

    # Network-stores (Neo4j/Qdrant) — точкуем в unreachable хост,
    # чтобы webui-init проглотил их при try/except и оставил None.
    # Тестам они не нужны, но без переопределения они полезут в localhost.
    monkeypatch.setenv("MEMORY__NEO4J_URI", "bolt://nowhere.invalid:7687")
    monkeypatch.setenv("MEMORY__QDRANT_URL", "http://nowhere.invalid:6333")

    # Сбрасываем cached singleton'ы — фабрики перечитают env при следующем
    # get_settings()/get_webui_settings(). Без этого тесты, выполненные
    # после первого create_app(), увидят старые значения.
    monkeypatch.setattr("harnes.config._settings", None)
    monkeypatch.setattr("harnes.webui.config._settings", None)
    return tmp_data_dir


@pytest.fixture
def webui_app(webui_env: Path) -> FastAPI:
    """ASGI-app без запущенного lifespan. Lifespan дёрнется в TestClient.__enter__."""
    from harnes.webui.app import create_app

    return create_app()


@pytest.fixture
def client(webui_app: FastAPI) -> Iterator[TestClient]:
    """TestClient как context-manager — гарантирует startup/shutdown."""
    with TestClient(webui_app) as c:
        yield c


@pytest.fixture
def goal_repo(webui_env: Path) -> GoalRepository:
    """GoalRepository по тому же пути, что и webui_app — для прямой проверки state."""
    from harnes.config import get_settings

    settings = get_settings()
    return GoalRepository(settings.goal_store.sqlite_path)


def _make_pending(
    description: str = "test pending",
    status: GoalStatus = GoalStatus.PENDING,
) -> Goal:
    return Goal(
        description=description,
        goal_class=GoalClass.TASK,
        predicate_of_success=JudgePredicate(criterion="ok if X"),
        status=status,
        origin=Origin.OPERATOR,
        originator="test",
    )


@pytest.fixture
def pending_goal(goal_repo: GoalRepository) -> Goal:
    """PENDING goal — для abandon / budget тестов."""
    g = _make_pending("test pending goal", GoalStatus.PENDING)
    goal_repo.create(g)
    return g


@pytest.fixture
def pending_approval_goal(goal_repo: GoalRepository) -> Goal:
    """PENDING_APPROVAL goal — для approve / reject тестов."""
    g = _make_pending("test pending-approval goal", GoalStatus.PENDING_APPROVAL)
    goal_repo.create(g)
    return g
