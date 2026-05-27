"""Cost dashboard — где утекают токены и латентность.

Research-критичный view: время серий по bucket'ам (часы/дни), разбивка по
step_type / skill_id / goal, top-10 goals by tokens, p50/p95 latency. Источник —
EpisodicStore (LanceDB): `recent_steps` (cost из самих шагов) + `recent_trajectories`
(skill_id живёт в trajectory.metadata_json).

Read-only. Никакого AJAX — всё рендерим server-side через query params (window,
group_by). Чарт — inline SVG (cream/accent палитра), без chart.js.

Масштаб: подтягиваем до 10_000 шагов / 2_000 траекторий и фильтруем в Python.
Для dev-объёмов хватает; на проде с миллионами шагов — TODO: pyarrow filter
push-down или предагрегация в отдельную таблицу.
"""
from __future__ import annotations

import json as jsonlib
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.memory.episodic import EpisodicStore
from harnes.webui.deps import get_episodic, get_goal_repo
from harnes.webui.templating import templates

router = APIRouter()


# ---- Constants -------------------------------------------------------------

_WINDOW_OPTIONS = ["1h", "24h", "7d", "30d", "all"]
_GROUP_BY_OPTIONS = ["hour", "day"]
_LLM_STEP_TYPES = frozenset({"thought", "action", "critique", "plan"})

# Дефолтная агрегация в зависимости от окна (если group_by не задан явно).
_DEFAULT_GROUP_BY: dict[str, str] = {
    "1h": "hour",
    "24h": "hour",
    "7d": "day",
    "30d": "day",
    "all": "day",
}

# Сколько bucket'ов рисуем в баре (последние N). Подобрано визуально под
# 600px ширину чарта.
_MAX_BUCKETS_RENDER = 60


# ---- Helpers ---------------------------------------------------------------


def _as_dt(value: Any) -> datetime | None:
    """LanceDB отдаёт naive datetime — нормализуем к aware UTC.

    None-safe: timestamp у trajectory.ended_at может быть None для running.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return None


def _window_to_timedelta(window: str) -> timedelta | None:
    """`"1h"|"24h"|"7d"|"30d"` → timedelta; `"all"` → None (без отсечения)."""
    if window == "all":
        return None
    if window == "1h":
        return timedelta(hours=1)
    if window == "24h":
        return timedelta(hours=24)
    if window == "7d":
        return timedelta(days=7)
    if window == "30d":
        return timedelta(days=30)
    raise HTTPException(400, f"window must be one of: {', '.join(_WINDOW_OPTIONS)}")


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile. Без numpy. q ∈ [0, 1]. Скопирован из
    eval.history._quantile, чтобы не зависеть от приватного имени.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _bucket_key(ts: datetime, group_by: str) -> datetime:
    """Truncate `ts` к началу bucket'а (hour/day) — стабильный sortable ключ."""
    if group_by == "hour":
        return ts.replace(minute=0, second=0, microsecond=0)
    # day
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def _bucket_label(bucket: datetime, group_by: str) -> str:
    """Короткая метка для оси X. Hour: `MM-DD HH`. Day: `MM-DD`."""
    if group_by == "hour":
        return bucket.strftime("%m-%d %H")
    return bucket.strftime("%m-%d")


def _enumerate_buckets(
    start: datetime, end: datetime, group_by: str
) -> list[datetime]:
    """Полный список bucket'ов от start до end включительно (с шагом hour/day).

    Гарантирует, что в чарте есть слоты для пустых периодов (бар height=0).
    Аккуратно с DST и месяцами: timedelta(hours/days) даёт ровные смещения.
    """
    step = timedelta(hours=1) if group_by == "hour" else timedelta(days=1)
    out: list[datetime] = []
    cur = _bucket_key(start, group_by)
    end_key = _bucket_key(end, group_by)
    # Safety cap — на огромном окне иначе уйдём в килобакеты.
    cap = 10_000
    while cur <= end_key and len(out) < cap:
        out.append(cur)
        cur = cur + step
    return out


# ---- Endpoint --------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
def cost_dashboard(
    request: Request,
    window: str = "24h",
    group_by: str | None = None,
    episodic: EpisodicStore = Depends(get_episodic),
    goal_repo=Depends(get_goal_repo),
) -> HTMLResponse:
    """HTML-страница cost dashboard. Все параметры — через query string."""
    # --- normalize params ---
    if window not in _WINDOW_OPTIONS:
        raise HTTPException(
            400, f"window must be one of: {', '.join(_WINDOW_OPTIONS)}"
        )
    if group_by is None or group_by == "":
        group_by = _DEFAULT_GROUP_BY[window]
    if group_by not in _GROUP_BY_OPTIONS:
        raise HTTPException(
            400, f"group_by must be one of: {', '.join(_GROUP_BY_OPTIONS)}"
        )

    delta = _window_to_timedelta(window)
    now = datetime.now(UTC)
    cutoff = (now - delta) if delta is not None else None

    # --- load raw data ---
    # TODO: pyarrow filter push-down если на проде объёмы вырастут — сейчас
    # 10_000 шагов / 2_000 траекторий вытягиваем целиком и фильтруем в Python.
    raw_steps = episodic.recent_steps(limit=10_000)
    raw_trajs = episodic.recent_trajectories(limit=2_000)

    # trajectory_id → trajectory dict (для goal_id и skill_id lookup).
    trajs_by_id: dict[str, dict[str, Any]] = {}
    for t in raw_trajs:
        tid = t.get("id")
        if tid:
            trajs_by_id[tid] = t

    # --- filter steps by window + only LLM-events (cost имеет смысл там) ---
    matched: list[dict[str, Any]] = []
    for r in raw_steps:
        st = r.get("step_type")
        if st not in _LLM_STEP_TYPES:
            continue
        ts = _as_dt(r.get("timestamp"))
        if ts is None:
            continue
        if cutoff is not None and ts < cutoff:
            continue
        # Лукап trajectory → skill_id.
        traj = trajs_by_id.get(r.get("trajectory_id") or "")
        skill_id = "unknown"
        if traj:
            try:
                meta = jsonlib.loads(traj.get("metadata_json") or "{}")
                if isinstance(meta, dict):
                    sid = meta.get("skill_id")
                    if isinstance(sid, str) and sid:
                        skill_id = sid
            except jsonlib.JSONDecodeError:
                pass
        matched.append(
            {
                "trajectory_id": r.get("trajectory_id"),
                "goal_id": r.get("goal_id"),
                "step_type": st,
                "timestamp": ts,
                "tokens": int(r.get("cost_tokens") or 0),
                "latency": float(r.get("cost_latency") or 0.0),
                "skill_id": skill_id,
            }
        )

    # --- global totals ---
    total_calls = len(matched)
    total_tokens = sum(s["tokens"] for s in matched)
    total_latency = sum(s["latency"] for s in matched)
    unique_goals = len({s["goal_id"] for s in matched if s["goal_id"]})
    unique_trajs = len({s["trajectory_id"] for s in matched if s["trajectory_id"]})

    latencies_sorted = sorted(s["latency"] for s in matched)
    p50 = _quantile(latencies_sorted, 0.5)
    p95 = _quantile(latencies_sorted, 0.95)

    # --- bucket aggregation (time series) ---
    # Определяем горизонт для bucket-слотов. Если window=all — от самого раннего
    # шага до now; иначе — от cutoff до now.
    if matched:
        earliest = min(s["timestamp"] for s in matched)
        latest = max(s["timestamp"] for s in matched)
    else:
        earliest = now
        latest = now
    horizon_start = cutoff if cutoff is not None else earliest
    horizon_end = max(latest, now)

    bucket_sums: dict[datetime, dict[str, float]] = {}
    for s in matched:
        key = _bucket_key(s["timestamp"], group_by)
        b = bucket_sums.setdefault(key, {"tokens": 0, "latency": 0.0, "calls": 0})
        b["tokens"] += s["tokens"]
        b["latency"] += s["latency"]
        b["calls"] += 1

    all_buckets = _enumerate_buckets(horizon_start, horizon_end, group_by)
    # Если bucket'ов больше лимита — берём хвост (последние). Для group_by=hour
    # и window=7d/30d это критично.
    if len(all_buckets) > _MAX_BUCKETS_RENDER:
        all_buckets = all_buckets[-_MAX_BUCKETS_RENDER:]

    bucket_rows: list[dict[str, Any]] = []
    max_bucket_tokens = 0
    for b in all_buckets:
        agg = bucket_sums.get(b, {"tokens": 0, "latency": 0.0, "calls": 0})
        tokens_v = int(agg["tokens"])
        bucket_rows.append(
            {
                "bucket": b,
                "label": _bucket_label(b, group_by),
                "tokens": tokens_v,
                "latency": float(agg["latency"]),
                "calls": int(agg["calls"]),
            }
        )
        if tokens_v > max_bucket_tokens:
            max_bucket_tokens = tokens_v

    # --- by step_type breakdown ---
    by_step: dict[str, dict[str, float]] = {
        t: {"tokens": 0, "calls": 0, "latency_sum": 0.0} for t in _LLM_STEP_TYPES
    }
    for s in matched:
        b = by_step[s["step_type"]]
        b["tokens"] += s["tokens"]
        b["calls"] += 1
        b["latency_sum"] += s["latency"]
    step_rows: list[dict[str, Any]] = []
    for stype, agg in by_step.items():
        calls = int(agg["calls"])
        step_rows.append(
            {
                "step_type": stype,
                "tokens": int(agg["tokens"]),
                "calls": calls,
                "mean_latency": (agg["latency_sum"] / calls) if calls else 0.0,
            }
        )
    # Сортировка для стабильности (tokens desc).
    step_rows.sort(key=lambda r: r["tokens"], reverse=True)
    max_step_tokens = max((r["tokens"] for r in step_rows), default=0)
    max_step_calls = max((r["calls"] for r in step_rows), default=0)

    # --- by skill_id breakdown ---
    by_skill: dict[str, dict[str, float]] = {}
    for s in matched:
        sid = s["skill_id"]
        b = by_skill.setdefault(sid, {"tokens": 0, "calls": 0, "latency_sum": 0.0})
        b["tokens"] += s["tokens"]
        b["calls"] += 1
        b["latency_sum"] += s["latency"]
    skill_rows: list[dict[str, Any]] = []
    for sid, agg in by_skill.items():
        calls = int(agg["calls"])
        skill_rows.append(
            {
                "skill_id": sid,
                "tokens": int(agg["tokens"]),
                "calls": calls,
                "mean_tokens": (agg["tokens"] / calls) if calls else 0.0,
                "mean_latency": (agg["latency_sum"] / calls) if calls else 0.0,
            }
        )
    skill_rows.sort(key=lambda r: r["tokens"], reverse=True)

    # --- top 10 goals by total tokens ---
    by_goal: dict[str, dict[str, int]] = {}
    for s in matched:
        gid = s["goal_id"]
        if not gid:
            continue
        b = by_goal.setdefault(gid, {"tokens": 0, "calls": 0})
        b["tokens"] += s["tokens"]
        b["calls"] += 1
    top_goals_raw = sorted(
        by_goal.items(), key=lambda kv: kv[1]["tokens"], reverse=True
    )[:10]
    # Лукапаем description'ы — fail-soft, если goal удалён.
    top_goals: list[dict[str, Any]] = []
    for gid, agg in top_goals_raw:
        desc = ""
        try:
            from uuid import UUID

            g = goal_repo.get(UUID(gid))
            if g is not None:
                desc = g.description[:80]
        except Exception:
            desc = ""
        top_goals.append(
            {
                "goal_id": gid,
                "description": desc,
                "tokens": agg["tokens"],
                "calls": agg["calls"],
            }
        )

    return templates.TemplateResponse(
        request,
        "cost/dashboard.html",
        {
            # filters
            "selected_window": window,
            "selected_group_by": group_by,
            "window_options": _WINDOW_OPTIONS,
            "group_by_options": _GROUP_BY_OPTIONS,
            # globals
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_latency": total_latency,
            "p50_latency": p50,
            "p95_latency": p95,
            "unique_goals": unique_goals,
            "unique_trajs": unique_trajs,
            # bucket chart
            "bucket_rows": bucket_rows,
            "max_bucket_tokens": max_bucket_tokens,
            # breakdowns
            "step_rows": step_rows,
            "max_step_tokens": max_step_tokens,
            "max_step_calls": max_step_calls,
            "skill_rows": skill_rows,
            "top_goals": top_goals,
            # для footer-подсказки
            "cutoff": cutoff,
            "now": now,
        },
    )
