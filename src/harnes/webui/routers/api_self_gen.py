"""Self-generation metrics dashboard — операционализация автономии.

Self-generation rate = ticks_with_self_spawn / total_ticks. Это доля тиков, на
которых агент **сам** породил себе цель (standing-policy / reflect inquiry /
proactive decomposition). См. § 1 архитектуры — "локус и зернистость контроля".

Источник — `TickJournal` (append-only events). Подтягиваем GOAL_SPAWNED'ы за
окно и сопоставляем с TICK_STARTED для знаменателя. GoalRepository нужен только
для подтягивания текущего статуса self-spawned целей (если ещё жива в repo).

Read-only, никаких мутаций.
"""
from __future__ import annotations

import json as jsonlib
from collections import Counter, OrderedDict, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.goals.store import GoalRepository
from harnes.metacycle.journal import TickEventType, TickJournal
from harnes.webui.deps import get_goal_repo, get_journal
from harnes.webui.templating import templates

router = APIRouter()


# ---------- window / bucket helpers ----------

_WINDOWS = OrderedDict(
    [
        ("1h", timedelta(hours=1)),
        ("24h", timedelta(hours=24)),
        ("7d", timedelta(days=7)),
        ("all", None),  # cutoff = None → since=None в recent_events
    ]
)

_GROUP_BY_OPTIONS = ("hour", "day")

# Защитный потолок на recent_events(limit=...) — пагинации нет, нужен явный bound.
# 50k событий покрывает >2 недель плотного run-loop (10s/tick * 50k ~ 5.7d тиков).
_MAX_FETCH = 50_000


def _bucket_dt(dt: datetime, group_by: str) -> datetime:
    """Нормализация timestamp к началу bucket'а (hour | day)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if group_by == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    # hour
    return dt.replace(minute=0, second=0, microsecond=0)


def _bucket_label(dt: datetime, group_by: str) -> str:
    """Короткая подпись для оси chart'а / таблицы."""
    if group_by == "day":
        return dt.strftime("%m-%d")
    return dt.strftime("%m-%d %H:00")


def _bucket_iso(dt: datetime, group_by: str) -> str:
    """Полная подпись для tooltip / aria-label."""
    if group_by == "day":
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d %H:00 UTC")


def _bucket_step(group_by: str) -> timedelta:
    return timedelta(days=1) if group_by == "day" else timedelta(hours=1)


def _iter_buckets(start: datetime, end: datetime, group_by: str):
    """Yield все bucket-boundaries от `start` до `end` включительно.

    Гарантирует, что пустые слоты есть в выводе (chart с провалом, а не
    отсутствием bar'а).
    """
    cur = _bucket_dt(start, group_by)
    last = _bucket_dt(end, group_by)
    step = _bucket_step(group_by)
    while cur <= last:
        yield cur
        cur = cur + step


def _safe_parse(payload_json: str) -> dict[str, Any]:
    try:
        v = jsonlib.loads(payload_json or "{}")
        return v if isinstance(v, dict) else {}
    except (jsonlib.JSONDecodeError, TypeError):
        return {}


def _to_uuid(value: Any) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


# ---------- page ----------


@router.get("", response_class=HTMLResponse)
def self_gen_dashboard(
    request: Request,
    window: str = "24h",
    group_by: str = "hour",
    journal: TickJournal = Depends(get_journal),
    goal_repo: GoalRepository = Depends(get_goal_repo),
) -> HTMLResponse:
    """Self-generation metrics dashboard.

    Computes:
      - total_ticks (TICK_STARTED count in window)
      - ticks_with_self_spawn (distinct tick_id among GOAL_SPAWNED in window)
      - total_spawned (raw GOAL_SPAWNED count in window)
      - self_generation_rate = ticks_with_self_spawn / total_ticks
      - per-bucket timeseries (для chart'а)
      - by-originator breakdown
      - by-goal_class breakdown
      - recent 20 spawned goals (с current status'ом из goal_repo если жива)
    """
    if window not in _WINDOWS:
        raise HTTPException(400, f"window must be one of: {', '.join(_WINDOWS)}")
    if group_by not in _GROUP_BY_OPTIONS:
        raise HTTPException(400, f"group_by must be one of: {', '.join(_GROUP_BY_OPTIONS)}")

    cutoff_delta = _WINDOWS[window]
    now = datetime.now(UTC)
    cutoff: datetime | None = (now - cutoff_delta) if cutoff_delta else None

    # --- numerator: GOAL_SPAWNED events в окне ---
    spawn_events = journal.recent_events(
        limit=_MAX_FETCH,
        event_type=TickEventType.GOAL_SPAWNED,
        since=cutoff,
    )

    # --- denominator: total ticks в том же окне ---
    # event_count(since) у TickJournal нет → fetch'аем TICK_STARTED и считаем len().
    # Лимит общий (_MAX_FETCH); если упёрлись — total_ticks занижен (флагуем hint'ом).
    tick_started_events = journal.recent_events(
        limit=_MAX_FETCH,
        event_type=TickEventType.TICK_STARTED,
        since=cutoff,
    )
    total_ticks = len(tick_started_events)
    truncated = total_ticks >= _MAX_FETCH

    # --- spawn-side aggregates ---
    total_spawned = len(spawn_events)
    ticks_with_self_spawn = len({e.tick_id for e in spawn_events})
    self_gen_rate = (
        (ticks_with_self_spawn / total_ticks * 100.0) if total_ticks > 0 else 0.0
    )

    # Per-bucket aggregation (counts of spawn events).
    per_bucket_counts: dict[datetime, int] = defaultdict(int)
    for ev in spawn_events:
        ts = ev.timestamp
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        per_bucket_counts[_bucket_dt(ts, group_by)] += 1

    # Always render the full window (с пустыми слотами), чтобы chart не "сжимался"
    # вокруг непустых.
    bucket_start = cutoff if cutoff is not None else (
        min((e.timestamp for e in spawn_events + tick_started_events if e.timestamp), default=now)
    )
    if bucket_start.tzinfo is None:
        bucket_start = bucket_start.replace(tzinfo=UTC)

    buckets: list[dict[str, Any]] = []
    max_bucket_count = 0
    for b in _iter_buckets(bucket_start, now, group_by):
        c = per_bucket_counts.get(b, 0)
        if c > max_bucket_count:
            max_bucket_count = c
        buckets.append(
            {
                "ts": b,
                "label": _bucket_label(b, group_by),
                "iso": _bucket_iso(b, group_by),
                "count": c,
            }
        )

    # --- by originator (groupby payload['originator']) ---
    originator_counts: Counter[str] = Counter()
    originator_examples: dict[str, list[str]] = defaultdict(list)
    by_goal_class: Counter[str] = Counter()
    for ev in spawn_events:
        payload = _safe_parse(ev.payload_json)
        orig = payload.get("originator") or "—"
        cls = payload.get("goal_class") or "—"
        originator_counts[str(orig)] += 1
        by_goal_class[str(cls)] += 1
        if len(originator_examples[str(orig)]) < 3:
            desc = payload.get("description") or ""
            if desc:
                originator_examples[str(orig)].append(str(desc)[:120])

    originator_rows: list[dict[str, Any]] = []
    for orig, count in originator_counts.most_common(10):
        pct = (count / total_spawned * 100.0) if total_spawned else 0.0
        originator_rows.append(
            {
                "originator": orig,
                "count": count,
                "pct": pct,
                "examples": originator_examples.get(orig, []),
            }
        )

    goal_class_rows: list[dict[str, Any]] = []
    max_class_count = max(by_goal_class.values(), default=0)
    for cls, count in by_goal_class.most_common():
        pct = (count / total_spawned * 100.0) if total_spawned else 0.0
        goal_class_rows.append(
            {
                "goal_class": cls,
                "count": count,
                "pct": pct,
                "bar_pct": (count / max_class_count * 100.0) if max_class_count else 0.0,
            }
        )

    # --- recent 20 spawned goals + current status ---
    # spawn_events уже отсортированы desc по id (recent_events).
    recent_rows: list[dict[str, Any]] = []
    for ev in spawn_events[:20]:
        payload = _safe_parse(ev.payload_json)
        goal_id = _to_uuid(payload.get("goal_id"))
        current_status: str | None = None
        goal_alive = False
        if goal_id is not None:
            try:
                goal = goal_repo.get(goal_id)
            except Exception:
                goal = None
            if goal is not None:
                goal_alive = True
                current_status = goal.status.value
        recent_rows.append(
            {
                "timestamp": ev.timestamp,
                "tick_id": ev.tick_id,
                "goal_id": str(goal_id) if goal_id is not None else None,
                "goal_id_short": (str(goal_id)[:8] if goal_id is not None else "—"),
                "goal_alive": goal_alive,
                "originator": payload.get("originator") or "—",
                "goal_class": payload.get("goal_class") or "—",
                "origin": payload.get("origin") or "—",
                "description": payload.get("description") or "",
                "current_status": current_status,
            }
        )

    return templates.TemplateResponse(
        request,
        "self_gen/dashboard.html",
        {
            # window state
            "selected_window": window,
            "selected_group_by": group_by,
            "all_windows": list(_WINDOWS.keys()),
            "all_group_by": list(_GROUP_BY_OPTIONS),
            "cutoff": cutoff,
            "now": now,
            # top stats
            "total_ticks": total_ticks,
            "ticks_with_self_spawn": ticks_with_self_spawn,
            "total_spawned": total_spawned,
            "self_gen_rate": self_gen_rate,
            "truncated": truncated,
            "max_fetch": _MAX_FETCH,
            # buckets / chart
            "buckets": buckets,
            "max_bucket_count": max_bucket_count,
            # breakdowns
            "originator_rows": originator_rows,
            "goal_class_rows": goal_class_rows,
            # recent
            "recent_rows": recent_rows,
        },
    )
