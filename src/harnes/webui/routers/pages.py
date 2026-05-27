"""Главные страницы: /, /dashboard, /health."""
from __future__ import annotations

import socket
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from harnes import AGENT_NAME
from harnes.config import Settings
from harnes.goals.schema import GoalStatus
from harnes.metacycle.journal import TickEventType
from harnes.webui.deps import (
    get_agent_settings,
)
from harnes.webui.templating import templates

router = APIRouter()


def _tcp_reachable(url: str, default_port: int, timeout: float = 1.0) -> bool:
    """Быстрый TCP-чек хоста из URL (http/bolt/grpc) — для health-panel."""
    try:
        if "://" in url:
            parsed = urlparse(url if url.startswith(("http", "bolt")) else f"http://{url}")
            host = parsed.hostname or url
            port = parsed.port or default_port
        else:
            host, _, p = url.partition(":")
            port = int(p) if p else default_port
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.gaierror, ValueError):
        return False


def _llm_reachable(api_base: str, timeout: float = 1.5) -> bool:
    """`is_router_reachable` уже есть — переиспользуем, но толерантно."""
    try:
        from harnes.llm import is_router_reachable

        return bool(is_router_reachable(timeout_s=timeout))
    except Exception:
        return False


@router.get("/", response_class=HTMLResponse)
def index() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=307)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    settings: Settings = Depends(get_agent_settings),
) -> HTMLResponse:
    """Главная: сводка по агенту — цели, последний tick, recent activity.

    Tolerant к недоступным stores — каждая секция рендерит свою заглушку,
    дашборд не падает целиком если что-то не подключилось.
    """
    state = request.app.state

    # Goals — может быть недоступен.
    by_status: dict[str, int] = {}
    active: list = []
    pending_approval: list = []
    if state.goal_repo is not None:
        for s in GoalStatus:
            by_status[s.value] = len(state.goal_repo.list_by_status(s))
        active = state.goal_repo.list_by_status(GoalStatus.ACTIVE)
        pending_approval = state.goal_repo.list_by_status(GoalStatus.PENDING_APPROVAL)

    # Journal — может быть недоступен.
    j_stats: dict = {"total_events": 0, "total_snapshots": 0, "by_event_type": {},
                     "min_tick_id": None, "max_tick_id": None}
    snap = None
    recent_events: list = []
    spawned_total = done_total = fail_total = 0
    if state.journal is not None:
        try:
            j_stats = state.journal.stats()
            snap = state.journal.latest_snapshot()
            recent_events = state.journal.recent_events(limit=15)
            spawned_total = state.journal.event_count(TickEventType.GOAL_SPAWNED)
            done_total = state.journal.event_count(TickEventType.GOAL_COMPLETED)
            fail_total = state.journal.event_count(TickEventType.GOAL_FAILED)
        except Exception:
            pass

    # Episodic — может быть недоступен.
    recent_trajs: list = []
    if state.episodic is not None:
        try:
            recent_trajs = state.episodic.recent_trajectories(limit=10)
        except Exception:
            pass

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "agent_name": AGENT_NAME,
            "settings": settings,
            "by_status": by_status,
            "active_goals": active,
            "pending_approval": pending_approval,
            "journal_stats": j_stats,
            "snapshot": snap,
            "recent_events": recent_events,
            "recent_trajs": recent_trajs,
            "spawned_total": spawned_total,
            "done_total": done_total,
            "fail_total": fail_total,
            "stores_status": {
                "goal_repo": state.goal_repo is not None,
                "journal": state.journal is not None,
                "episodic": state.episodic is not None,
            },
        },
    )


@router.get("/health", response_class=HTMLResponse)
def health(
    request: Request,
    settings: Settings = Depends(get_agent_settings),
) -> HTMLResponse:
    """Health-страница: статус всех backend'ов."""
    qdrant_ok = _tcp_reachable(settings.memory.qdrant_url, 6333)
    neo4j_ok = _tcp_reachable(settings.memory.neo4j_uri, 7687)
    llm_ok = _llm_reachable(settings.llm.api_base)

    # Файловые stores
    sqlite_goal_ok = settings.goal_store.sqlite_path.parent.exists()
    sqlite_journal_ok = settings.metacycle.journal_db_path.parent.exists()
    lancedb_ok = settings.memory.lancedb_path.exists()
    skills_ok = settings.procedural_store.bundles_dir.exists()

    state = request.app.state
    return templates.TemplateResponse(
        request,
        "health.html",
        {
            "settings": settings,
            "backends": [
                {"name": "Goal store (SQLite)", "ok": sqlite_goal_ok,
                 "target": str(settings.goal_store.sqlite_path)},
                {"name": "Tick journal (SQLite)", "ok": sqlite_journal_ok,
                 "target": str(settings.metacycle.journal_db_path)},
                {"name": "Episodic (LanceDB)", "ok": lancedb_ok,
                 "target": str(settings.memory.lancedb_path)},
                {"name": "Skills (bundles dir)", "ok": skills_ok,
                 "target": str(settings.procedural_store.bundles_dir)},
                {"name": "Semantic (Qdrant)", "ok": qdrant_ok,
                 "target": settings.memory.qdrant_url},
                {"name": "World model (Neo4j+Graphiti)", "ok": neo4j_ok,
                 "target": settings.memory.neo4j_uri},
                {"name": "LLM router", "ok": llm_ok,
                 "target": settings.llm.api_base},
            ],
            "stores": {
                "goal_repo": state.goal_repo is not None,
                "episodic": state.episodic is not None,
                "journal": state.journal is not None,
                "skill_registry": state.skill_registry is not None,
                "semantic": state.semantic is not None,
                "world": state.world is not None,
            },
        },
    )
