"""Dependency providers для FastAPI routers.

Stores кэшируются в `app.state` — один объект на процесс. Поля могут быть
None, если backend недоступен (Neo4j/Qdrant down) — роутеры обязаны это
проверять и показывать заглушку, а не ронять страницу.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import HTTPException, Request

from harnes.config import Settings
from harnes.eval import EvalHistoryStore
from harnes.goals.store import GoalRepository
from harnes.memory.episodic import EpisodicStore
from harnes.memory.router import MemoryRouter
from harnes.memory.semantic import SemanticStore
from harnes.memory.world import WorldModelStore
from harnes.metacycle.commands import CommandStore
from harnes.metacycle.journal import TickJournal
from harnes.skills.store import SkillRegistry

log = structlog.get_logger()


# ---------- accessors (read app.state) ----------


def get_agent_settings(request: Request) -> Settings:
    return request.app.state.agent_settings  # type: ignore[no-any-return]


def get_goal_repo(request: Request) -> GoalRepository:
    repo = request.app.state.goal_repo
    if repo is None:
        raise HTTPException(503, "goal_store недоступен — проверь settings.goal_store.sqlite_path")
    return repo  # type: ignore[no-any-return]


def get_episodic(request: Request) -> EpisodicStore:
    ep = request.app.state.episodic
    if ep is None:
        raise HTTPException(503, "episodic store недоступен (LanceDB не открыт)")
    return ep  # type: ignore[no-any-return]


def get_journal(request: Request) -> TickJournal:
    j = request.app.state.journal
    if j is None:
        raise HTTPException(503, "tick journal недоступен")
    return j  # type: ignore[no-any-return]


def get_skill_registry(request: Request) -> SkillRegistry:
    sr = request.app.state.skill_registry
    if sr is None:
        raise HTTPException(503, "skill registry недоступен")
    return sr  # type: ignore[no-any-return]


def get_eval_history(request: Request) -> EvalHistoryStore:
    eh = request.app.state.eval_history
    if eh is None:
        raise HTTPException(503, "eval history недоступен")
    return eh  # type: ignore[no-any-return]


def get_command_store(request: Request) -> CommandStore:
    cs = request.app.state.command_store
    if cs is None:
        raise HTTPException(503, "command store недоступен (path не writable)")
    return cs  # type: ignore[no-any-return]


def get_memory_router(request: Request) -> MemoryRouter:
    """MemoryRouter с теми из {episodic, semantic, world} что доступны.

    Дёргается лениво — semantic/world могут отсутствовать, это не ошибка.
    """
    if not hasattr(request.app.state, "memory_router") or request.app.state.memory_router is None:
        request.app.state.memory_router = MemoryRouter(
            episodic=request.app.state.episodic,
            semantic=request.app.state.semantic,
            world=request.app.state.world,
        )
    return request.app.state.memory_router  # type: ignore[no-any-return]


def get_world(request: Request) -> WorldModelStore | None:
    """Опциональный — может быть None если Neo4j недоступен."""
    return request.app.state.world  # type: ignore[no-any-return]


def get_semantic(request: Request) -> SemanticStore | None:
    return request.app.state.semantic  # type: ignore[no-any-return]


# ---------- init helpers (вызываются из lifespan) ----------


def _parent_writable(p: Path) -> bool:
    """Каталог-родитель существует ИЛИ может быть создан без падения."""
    try:
        parent = Path(p).parent
        if parent.exists():
            return True
        parent.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def init_stores(settings: Settings) -> dict[str, Any]:
    """Инициализирует все stores. Тяжёлые/сетевые — try/except, None при ошибке.

    Возвращает dict для записи в app.state.
    """
    state: dict[str, Any] = {
        "agent_settings": settings,
        "goal_repo": None,
        "episodic": None,
        "journal": None,
        "skill_registry": None,
        "eval_history": None,
        "semantic": None,
        "world": None,
        "memory_router": None,
        "command_store": None,
    }

    # --- Lightweight (file/sqlite, no network) ---
    # Каждый init требует существующего родительского каталога. Если data/
    # не примонтирован — оставляем None, чтобы dashboard не строил мусорный
    # in-memory fallback (TickJournal делает sqlite:///:memory: который ломается
    # на следующем Session — каждый коннект получает новый пустой DB).
    if _parent_writable(settings.goal_store.sqlite_path):
        try:
            state["goal_repo"] = GoalRepository(settings.goal_store.sqlite_path)
        except Exception as exc:
            log.warning("webui.init.goal_repo.failed", error=str(exc))
    else:
        log.warning("webui.init.goal_repo.path_missing",
                    path=str(settings.goal_store.sqlite_path))

    if _parent_writable(settings.memory.lancedb_path):
        try:
            settings.memory.lancedb_path.mkdir(parents=True, exist_ok=True)
            state["episodic"] = EpisodicStore(settings.memory.lancedb_path)
        except Exception as exc:
            log.warning("webui.init.episodic.failed", error=str(exc))
    else:
        log.warning("webui.init.episodic.path_missing",
                    path=str(settings.memory.lancedb_path))

    if _parent_writable(settings.metacycle.journal_db_path):
        try:
            state["journal"] = TickJournal(settings.metacycle.journal_db_path)
        except Exception as exc:
            log.warning("webui.init.journal.failed", error=str(exc))
    else:
        log.warning("webui.init.journal.path_missing",
                    path=str(settings.metacycle.journal_db_path))

    if settings.procedural_store.bundles_dir.exists():
        try:
            state["skill_registry"] = SkillRegistry(
                settings.procedural_store.bundles_dir,
                settings.procedural_store.sqlite_path,
            )
        except Exception as exc:
            log.warning("webui.init.skills.failed", error=str(exc))
    else:
        log.warning("webui.init.skills.path_missing",
                    path=str(settings.procedural_store.bundles_dir))

    if _parent_writable(settings.eval.history_db_path):
        try:
            state["eval_history"] = EvalHistoryStore(settings.eval.history_db_path)
        except Exception as exc:
            log.warning("webui.init.eval_history.failed", error=str(exc))

    if _parent_writable(settings.metacycle.commands_db_path):
        try:
            state["command_store"] = CommandStore(settings.metacycle.commands_db_path)
        except Exception as exc:
            log.warning("webui.init.command_store.failed", error=str(exc))
    else:
        log.warning("webui.init.command_store.path_missing",
                    path=str(settings.metacycle.commands_db_path))

    # --- Network-зависимые: пытаемся, но не валим страницу при отказе ---
    try:
        state["semantic"] = SemanticStore(settings.memory.qdrant_url)
    except Exception as exc:
        log.warning("webui.init.semantic.failed", error=str(exc))

    try:
        state["world"] = WorldModelStore(
            settings.memory.neo4j_uri,
            settings.memory.neo4j_user,
            settings.memory.neo4j_password,
        )
    except Exception as exc:
        log.warning("webui.init.world.failed", error=str(exc))

    return state


def close_stores(state: dict[str, Any]) -> None:
    """Закрытие соединений на shutdown."""
    world = state.get("world")
    if world is not None:
        try:
            world.close()
        except Exception as exc:
            log.warning("webui.close.world.failed", error=str(exc))
