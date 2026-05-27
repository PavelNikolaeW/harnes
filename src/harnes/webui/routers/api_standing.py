"""Standing-goals dashboard — реактивный слой (§ 5 архитектуры).

Standing-goals — это `GoalClass.STANDING` policies (status=ACTIVE), которые сами
не выполняются; на каждом тике их `policy_name` callable проверяется, и при
срабатывании условия порождается дочерний task-goal (parent_id = standing.id).

Маркировка spawn'а: см. `harnes/metacycle/standing.py` — `originator=f"standing:{parent.id}"`
для всех spawn'ов через policy callable. CLI append'ит GOAL_SPAWNED events с
этим originator'ом в `harnes/operator/cli.py` (run-loop). Соответственно мэтчинг
spawn → standing идёт по точному строковому матчу `f"standing:{sg.id}"`.

Read-only, без мутаций.
"""
from __future__ import annotations

import json as jsonlib
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from harnes.goals.schema import GoalClass, GoalStatus
from harnes.goals.store import GoalRepository
from harnes.metacycle.journal import TickEventType, TickJournal
from harnes.webui.deps import get_goal_repo, get_journal
from harnes.webui.templating import templates

router = APIRouter()


# ---------- constants ----------

# Дефолтное окно для timeline (нижний chart). Top-stats считаются all-time.
_TIMELINE_WINDOW = timedelta(days=7)
# Окно для "fires last 24h" stat card.
_RECENT_WINDOW = timedelta(hours=24)
# Защитный потолок выборки. recent_events не имеет пагинации.
_MAX_FETCH = 50_000
# Скольких последних spawn'ов в per-policy "recent spawns".
_RECENT_PER_POLICY = 3
# Скольких детей подтягивать как "child goals (most recent)" в effectiveness расчёт.
_CHILDREN_LIMIT = 20


# ---------- helpers ----------


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


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _bucket_hour(dt: datetime) -> datetime:
    dt = _ensure_utc(dt) or datetime.now(UTC)
    return dt.replace(minute=0, second=0, microsecond=0)


def _iter_hours(start: datetime, end: datetime):
    cur = _bucket_hour(start)
    last = _bucket_hour(end)
    step = timedelta(hours=1)
    while cur <= last:
        yield cur
        cur = cur + step


# ---------- dashboard route ----------


@router.get("", response_class=HTMLResponse)
def standing_dashboard(
    request: Request,
    journal: TickJournal = Depends(get_journal),
    goal_repo: GoalRepository = Depends(get_goal_repo),
) -> HTMLResponse:
    """Standing-goals effectiveness dashboard.

    Для каждой STANDING-цели собирает:
      - spawn count (fires all-time) — GOAL_SPAWNED events с originator=f"standing:{id}"
      - first/last fire timestamps
      - children: текущие статусы (done/failed/active/pending/abandoned/gone)
        Источник — GOAL_SPAWNED events (payload.goal_id), затем `goal_repo.get(id)`
        для current status. Если goal удалён из repo → помечается "gone".
      - effectiveness = done_children / total_spawned (если spawns==0 → None)

    Глобально:
      - count standing-целей (active vs all)
      - total fires all-time
      - fires в last 24h
      - mean effectiveness по всем active standing-целям с ≥1 spawn
      - 7d hourly spawn-timeline (для bottom-chart)
    """
    now = datetime.now(UTC)
    timeline_cutoff = now - _TIMELINE_WINDOW
    recent_cutoff = now - _RECENT_WINDOW

    # --- load standing goals ---
    standing_goals = goal_repo.list_by_class(GoalClass.STANDING)
    active_count = sum(1 for g in standing_goals if g.status == GoalStatus.ACTIVE)

    # --- load ALL GOAL_SPAWNED events (all-time, capped) ---
    # event_count() per-originator не поддерживается → fetch'аем и фильтруем in-mem.
    spawn_events_all = journal.recent_events(
        limit=_MAX_FETCH,
        event_type=TickEventType.GOAL_SPAWNED,
    )
    truncated = len(spawn_events_all) >= _MAX_FETCH

    # --- индексация spawn'ов по originator ---
    # originator format в standing.py: f"standing:{parent.id}"
    # Building per-originator-string index → matches by exact string.
    spawns_by_originator: dict[str, list[Any]] = defaultdict(list)
    total_standing_fires_all_time = 0
    fires_recent_24h = 0
    timeline_counts: dict[datetime, int] = defaultdict(int)
    for ev in spawn_events_all:
        payload = _safe_parse(ev.payload_json)
        orig = payload.get("originator") or ""
        if not isinstance(orig, str) or not orig.startswith("standing:"):
            continue
        spawns_by_originator[orig].append((ev, payload))
        total_standing_fires_all_time += 1
        ts = _ensure_utc(ev.timestamp)
        if ts is not None:
            if ts >= recent_cutoff:
                fires_recent_24h += 1
            if ts >= timeline_cutoff:
                timeline_counts[_bucket_hour(ts)] += 1

    # --- per-policy aggregation ---
    policy_rows: list[dict[str, Any]] = []
    effectiveness_values: list[float] = []

    for sg in sorted(
        standing_goals, key=lambda g: (g.status.value, -g.priority, g.created_at)
    ):
        key = f"standing:{sg.id}"
        spawns = spawns_by_originator.get(key, [])
        # spawns пришли desc (recent_events sort by id desc) — оставляем как есть.
        spawn_count = len(spawns)

        # first/last fire times
        first_fire: datetime | None = None
        last_fire: datetime | None = None
        if spawns:
            tss = [_ensure_utc(ev.timestamp) for ev, _ in spawns]
            tss = [t for t in tss if t is not None]
            if tss:
                first_fire = min(tss)
                last_fire = max(tss)

        # children → current status. Подтягиваем child goal_id из payload
        # (НЕ через list_children — там могли остаться unrelated дети если standing.id
        # пере-присваивали; payload — source of truth для конкретного spawn).
        child_status_counts: dict[str, int] = defaultdict(int)
        gone_count = 0
        for _ev, payload in spawns:
            gid = _to_uuid(payload.get("goal_id"))
            if gid is None:
                child_status_counts["unknown"] += 1
                continue
            try:
                g = goal_repo.get(gid)
            except Exception:
                g = None
            if g is None:
                # spawned ребёнок удалён из repo (clean run / retention)
                # → помечается "gone", в effectiveness не учитывается ни как done, ни как fail.
                gone_count += 1
                child_status_counts["gone"] += 1
            else:
                child_status_counts[g.status.value] += 1

        # effectiveness: done / total_spawned. "gone" в знаменателе остаётся
        # (отражает реальный % успехов от всех fire'ов; gone скорее всего был done
        # до retention, но мы не можем это утверждать).
        done_count = child_status_counts.get(GoalStatus.DONE.value, 0)
        if spawn_count > 0:
            eff_pct = done_count / spawn_count * 100.0
            effectiveness_values.append(eff_pct)
        else:
            eff_pct = None

        # recent spawns (top N) с current status
        recent_spawns: list[dict[str, Any]] = []
        for ev, payload in spawns[:_RECENT_PER_POLICY]:
            gid = _to_uuid(payload.get("goal_id"))
            status_str: str | None = None
            if gid is not None:
                try:
                    g = goal_repo.get(gid)
                except Exception:
                    g = None
                if g is not None:
                    status_str = g.status.value
                else:
                    status_str = "gone"
            recent_spawns.append(
                {
                    "timestamp": _ensure_utc(ev.timestamp),
                    "tick_id": ev.tick_id,
                    "goal_id": str(gid) if gid else None,
                    "goal_id_short": str(gid)[:8] if gid else "—",
                    "description": (payload.get("description") or "")[:120],
                    "status": status_str,
                }
            )

        policy_name = (
            sg.metadata.get("policy_name") if isinstance(sg.metadata, dict) else None
        )

        policy_rows.append(
            {
                "id": str(sg.id),
                "id_short": str(sg.id)[:8],
                "description": sg.description,
                "status": sg.status.value,
                "priority": sg.priority,
                "policy_name": policy_name or "—",
                "spawn_count": spawn_count,
                "first_fire": first_fire,
                "last_fire": last_fire,
                "eff_pct": eff_pct,
                "child_status_counts": dict(child_status_counts),
                "done_count": done_count,
                "gone_count": gone_count,
                "recent_spawns": recent_spawns,
            }
        )

    mean_eff: float | None = (
        sum(effectiveness_values) / len(effectiveness_values)
        if effectiveness_values
        else None
    )

    # --- timeline buckets за last 7d (hourly) ---
    buckets: list[dict[str, Any]] = []
    max_bucket_count = 0
    if standing_goals:
        for b in _iter_hours(timeline_cutoff, now):
            c = timeline_counts.get(b, 0)
            if c > max_bucket_count:
                max_bucket_count = c
            buckets.append(
                {
                    "ts": b,
                    "label": b.strftime("%m-%d %H:00"),
                    "iso": b.strftime("%Y-%m-%d %H:00 UTC"),
                    "count": c,
                }
            )

    return templates.TemplateResponse(
        request,
        "standing/dashboard.html",
        {
            # top stats
            "standing_count": len(standing_goals),
            "standing_active_count": active_count,
            "total_fires_all_time": total_standing_fires_all_time,
            "fires_recent_24h": fires_recent_24h,
            "mean_eff_pct": mean_eff,
            "truncated": truncated,
            "max_fetch": _MAX_FETCH,
            # per-policy
            "policy_rows": policy_rows,
            # timeline
            "buckets": buckets,
            "max_bucket_count": max_bucket_count,
            "timeline_cutoff": timeline_cutoff,
            "now": now,
        },
    )
