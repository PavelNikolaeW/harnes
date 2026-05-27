"""Web→agent IPC: pause/resume/trigger + история команд.

См. harnes/metacycle/commands.py + run-loop в operator/cli.py.

Команды складываются в append-only log; agent drain'ит их в начале каждой
итерации. Webui не дожидается выполнения — UI оптимистично показывает
сообщение и возвращает на страницу команд.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from harnes.metacycle.commands import CommandStore, CommandType
from harnes.webui.config import WebuiSettings, get_webui_settings
from harnes.webui.deps import get_command_store
from harnes.webui.templating import templates

router = APIRouter()


def _is_loop_paused(store: CommandStore) -> bool:
    """Текущее состояние loop — по последней consumed pause/resume команде.

    Survivable: если оператор поставил pause, перезапустил контейнер, и run-loop
    подтянул pause-state из CommandStore — UI это покажет корректно.
    """
    try:
        return store.latest_pause_state(only_consumed=True)
    except Exception:
        return False


@router.get("", response_class=HTMLResponse)
def list_commands(
    request: Request,
    store: CommandStore = Depends(get_command_store),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> HTMLResponse:
    """История последних команд + быстрая панель управления."""
    rows = store.recent(limit=100)
    unconsumed = store.count_unconsumed()
    paused = _is_loop_paused(store)
    return templates.TemplateResponse(
        request,
        "commands/list.html",
        {
            "rows": rows,
            "unconsumed": unconsumed,
            "paused": paused,
            "read_only": cfg.read_only,
            "command_types": [c.value for c in CommandType],
        },
    )


def _issue_and_redirect(
    request: Request,
    command: CommandType,
    store: CommandStore,
    cfg: WebuiSettings,
) -> RedirectResponse:
    if cfg.read_only:
        raise HTTPException(403, "read-only mode")
    store.issue(command, issuer="webui")
    # Возврат на referer если он внутри webui — иначе на /commands.
    ref = request.headers.get("referer", "/commands")
    return RedirectResponse(url=ref, status_code=303)


@router.post("/pause", response_class=HTMLResponse)
def pause(
    request: Request,
    store: CommandStore = Depends(get_command_store),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    return _issue_and_redirect(request, CommandType.PAUSE, store, cfg)


@router.post("/resume", response_class=HTMLResponse)
def resume(
    request: Request,
    store: CommandStore = Depends(get_command_store),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    return _issue_and_redirect(request, CommandType.RESUME, store, cfg)


@router.post("/trigger_tick", response_class=HTMLResponse)
def trigger_tick(
    request: Request,
    store: CommandStore = Depends(get_command_store),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    return _issue_and_redirect(request, CommandType.TRIGGER_TICK, store, cfg)
