"""FastAPI application factory для webui.

См. `src/harnes/webui/__init__.py` и `webui/README.md`.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from harnes import AGENT_NAME, PROJECT_NAME, __version__
from harnes.config import get_settings
from harnes.telemetry import setup_logging
from harnes.webui.config import get_webui_settings
from harnes.webui.deps import close_stores, init_stores
from harnes.webui.routers import (
    api_commands,
    api_cost,
    api_diff,
    api_eval,
    api_goals,
    api_journal,
    api_llm,
    api_memory,
    api_reflect,
    api_replay,
    api_self_gen,
    api_skills,
    api_trajectories,
    pages,
)

log = structlog.get_logger()

_PKG_DIR = Path(__file__).parent
STATIC_DIR = _PKG_DIR / "static"
TEMPLATES_DIR = _PKG_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: открываем stores. Shutdown: закрываем."""
    settings = get_settings()
    state = init_stores(settings)
    for key, value in state.items():
        setattr(app.state, key, value)

    log.info(
        "webui.started",
        agent=AGENT_NAME,
        version=__version__,
        goal_repo=bool(state["goal_repo"]),
        episodic=bool(state["episodic"]),
        journal=bool(state["journal"]),
        semantic=bool(state["semantic"]),
        world=bool(state["world"]),
    )
    try:
        yield
    finally:
        close_stores(state)
        log.info("webui.stopped")


def create_app() -> FastAPI:
    """ASGI factory — FastAPI app с router'ами и static."""
    webui_cfg = get_webui_settings()
    setup_logging(webui_cfg.log_level)

    app = FastAPI(
        title=f"{PROJECT_NAME} · {AGENT_NAME}",
        description=(
            "Admin-консоль для исследовательского автономного агента. "
            "Read-only observability + минимальное управление целями."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # Mount static.
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Routers.
    app.include_router(pages.router)
    app.include_router(api_goals.router, prefix="/goals", tags=["goals"])
    app.include_router(api_trajectories.router, prefix="/trajectories", tags=["trajectories"])
    app.include_router(api_journal.router, prefix="/journal", tags=["journal"])
    app.include_router(api_memory.router, prefix="/memory", tags=["memory"])
    app.include_router(api_skills.router, prefix="/skills", tags=["skills"])
    app.include_router(api_eval.router, prefix="/eval", tags=["eval"])
    app.include_router(api_commands.router, prefix="/commands", tags=["commands"])
    app.include_router(api_llm.router, prefix="/llm", tags=["llm"])
    app.include_router(api_replay.router, prefix="/replay", tags=["replay"])
    app.include_router(api_diff.router, prefix="/trajectories-diff", tags=["diff"])
    app.include_router(api_cost.router, prefix="/cost", tags=["cost"])
    app.include_router(api_self_gen.router, prefix="/self-gen", tags=["self-gen"])
    app.include_router(api_reflect.router, prefix="/reflect", tags=["reflect"])

    return app


# Module-level ASGI app для production (uvicorn harnes.webui.app:app).
app = create_app()
