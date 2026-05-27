"""Reflect events explorer — таймлайн skill_versioned + inquiry_spawned.

Reflect — триггерный этап метацикла (§ 15 архитектуры). Запускается при
verify=FAIL и опционально на других триггерах (volume, budget_exceeded,
prediction_divergence, novelty, schedule).

Источники событий:
- ``SKILL_VERSIONED`` — failure_analysis bump'нул prompt_template скилла.
- ``GOAL_SPAWNED`` с ``payload['originator']`` начинающимся на ``reflect.`` —
  inquiry_from_failure spawn'ил inquiry-goal.

Все события мерджатся в один timeline (sorted by timestamp desc).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.goals.store import GoalRepository
from harnes.metacycle.journal import TickEventRow, TickEventType, TickJournal
from harnes.skills.store import SkillRegistry
from harnes.webui.deps import get_goal_repo, get_journal, get_skill_registry
from harnes.webui.templating import templates

log = structlog.get_logger()

router = APIRouter()


# ---------- Windows ----------

_WINDOWS: dict[str, timedelta | None] = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "all": None,
}


def _window_cutoff(window: str) -> datetime | None:
    """Возвращает datetime cutoff для recent_events(since=...). None = no filter."""
    delta = _WINDOWS.get(window)
    if delta is None:
        return None
    return datetime.now(UTC) - delta


# ---------- Payload parsing ----------


def _parse_payload(row: TickEventRow) -> dict[str, Any]:
    """Безопасный парс payload_json → dict."""
    raw = row.payload_json or "{}"
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return {"_raw": data}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


def _is_reflect_originator(originator: str | None) -> bool:
    """GOAL_SPAWNED от reflect различаем по originator-prefix.

    См. ``harnes.metacycle.reflect.reflect_inquiry_from_failure`` —
    она ставит ``originator=f"reflect.inquiry_from_failure:{goal.id}"``.
    Префикс ``reflect.`` отделяет от standing/bootstrap/webui/cli.
    """
    if not originator:
        return False
    return originator.startswith("reflect.") or originator == "reflect"


# ---------- Mean time between ----------


def _mean_seconds_between(timestamps: list[datetime]) -> float | None:
    """Среднее расстояние между соседними events (по факту = диапазон / (n-1)).

    Простая метрика: если в окне >=2 events, считаем (max-min)/(n-1).
    None если <2 events.
    """
    if len(timestamps) < 2:
        return None
    sorted_ts = sorted(timestamps)
    span = (sorted_ts[-1] - sorted_ts[0]).total_seconds()
    return span / (len(sorted_ts) - 1)


def _format_seconds(seconds: float | None) -> str:
    """'1.4h between events' / '23m between events' / '—'."""
    if seconds is None:
        return "—"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    if seconds < 5400 * 24:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# ---------- Route ----------


@router.get("", response_class=HTMLResponse)
def reflect_explorer(
    request: Request,
    window: str = "7d",
    mode: str = "all",
    journal: TickJournal = Depends(get_journal),
    skill_registry: SkillRegistry = Depends(get_skill_registry),
    goal_repo: GoalRepository = Depends(get_goal_repo),
) -> HTMLResponse:
    """Timeline reflect-событий за выбранное окно.

    Query params:
        window: "1h" | "24h" | "7d" | "all" (default 7d — reflect-rate низкий).
        mode:   "skill_versioned" | "inquiry_spawned" | "all" (default all).
    """
    if window not in _WINDOWS:
        raise HTTPException(400, f"unknown window: {window} (use 1h|24h|7d|all)")
    if mode not in ("all", "skill_versioned", "inquiry_spawned"):
        raise HTTPException(400, f"unknown mode: {mode}")

    cutoff = _window_cutoff(window)

    # ---------- Load events ----------
    skill_rows: list[TickEventRow] = []
    if mode in ("all", "skill_versioned"):
        skill_rows = journal.recent_events(
            limit=500,
            event_type=TickEventType.SKILL_VERSIONED,
            since=cutoff,
        )

    inquiry_rows: list[TickEventRow] = []
    if mode in ("all", "inquiry_spawned"):
        all_spawned = journal.recent_events(
            limit=500,
            event_type=TickEventType.GOAL_SPAWNED,
            since=cutoff,
        )
        for row in all_spawned:
            payload = _parse_payload(row)
            if _is_reflect_originator(payload.get("originator")):
                inquiry_rows.append(row)

    # ---------- Build unified items ----------
    items: list[dict[str, Any]] = []

    for row in skill_rows:
        payload = _parse_payload(row)
        skill_id = payload.get("skill_id")
        skill_obj = None
        if skill_id:
            try:
                skill_obj = skill_registry.get(str(skill_id))
            except Exception as exc:
                log.debug("reflect.skill_lookup.failed", skill_id=skill_id, error=str(exc))
        items.append(
            {
                "mode": "skill_versioned",
                "timestamp": row.timestamp,
                "tick_id": row.tick_id,
                "payload": payload,
                "skill_id": skill_id,
                "skill": skill_obj,
                "from_version": payload.get("from_version"),
                "to_version": payload.get("to_version"),
                "rationale": payload.get("diagnosis")
                or payload.get("rationale")
                or payload.get("reason"),
            }
        )

    for row in inquiry_rows:
        payload = _parse_payload(row)
        goal_id_raw = payload.get("goal_id") or payload.get("inquiry_id")
        goal_obj = None
        if goal_id_raw:
            try:
                goal_obj = goal_repo.get(UUID(str(goal_id_raw)))
            except (ValueError, TypeError):
                pass
            except Exception as exc:
                log.debug("reflect.goal_lookup.failed", goal_id=goal_id_raw, error=str(exc))
        # parent_goal_id может быть в payload (как parent_goal_id) либо
        # на самом Goal (как parent_id). Из originator вытаскиваем тоже —
        # формат "reflect.inquiry_from_failure:{parent_uuid}".
        parent_goal_id = payload.get("parent_goal_id") or payload.get("parent_id")
        if not parent_goal_id and goal_obj is not None:
            parent_goal_id = str(goal_obj.parent_id) if goal_obj.parent_id else None
        if not parent_goal_id:
            originator = payload.get("originator") or ""
            if ":" in originator:
                parent_goal_id = originator.split(":", 1)[1]

        items.append(
            {
                "mode": "inquiry_spawned",
                "timestamp": row.timestamp,
                "tick_id": row.tick_id,
                "payload": payload,
                "goal_id": str(goal_id_raw) if goal_id_raw else None,
                "goal": goal_obj,
                "description": (goal_obj.description if goal_obj else payload.get("description"))
                or "(no description)",
                "parent_goal_id": parent_goal_id,
                "originator": payload.get("originator"),
            }
        )

    items.sort(
        key=lambda i: i["timestamp"] or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )

    # ---------- Stats ----------
    skill_count = sum(1 for i in items if i["mode"] == "skill_versioned")
    inquiry_count = sum(1 for i in items if i["mode"] == "inquiry_spawned")
    total = len(items)

    all_ts = [i["timestamp"] for i in items if i["timestamp"] is not None]
    mean_between = _mean_seconds_between(all_ts)
    last_ts = max(all_ts) if all_ts else None

    stats = {
        "total": total,
        "skill_versioned": skill_count,
        "inquiry_spawned": inquiry_count,
        "mean_between_human": _format_seconds(mean_between),
        "last_ts": last_ts,
    }

    return templates.TemplateResponse(
        request,
        "reflect/explorer.html",
        {
            "items": items,
            "stats": stats,
            "window": window,
            "mode": mode,
            "windows": list(_WINDOWS.keys()),
            "modes": ["all", "skill_versioned", "inquiry_spawned"],
        },
    )
