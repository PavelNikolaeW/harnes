"""LLM Call Trace Viewer — aggregate-view over episodic steps as LLM events.

Каждый thought/action/critique/plan-шаг в траектории = один LLM-вызов; у него есть
cost_tokens + cost_latency. Полный prompt/response мы не сохраняем (см. § 7 архитектуры) —
этот view агрегирует уже существующие данные шагов в события.

Read-only. Источник — EpisodicStore (LanceDB). LanceDB filter DSL ограничен, поэтому
вытягиваем widely и фильтруем в Python — для нашего масштаба (десятки-сотни тысяч
шагов) этого достаточно. Если упрёмся — переключиться на pyarrow filter expressions.
"""
from __future__ import annotations

import json as jsonlib
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.memory.episodic import EpisodicStore
from harnes.webui.deps import get_episodic
from harnes.webui.templating import templates

router = APIRouter()


# LLM-вызовы порождаются именно этими типами шагов; observation/retry_note —
# не LLM-events (это tool-result или meta-note).
_LLM_STEP_TYPES = frozenset({"thought", "action", "critique", "plan"})
_ALL_FILTER_OPTIONS = ["all", "thought", "action", "critique", "plan"]


def _extract_model(content: dict[str, Any]) -> str:
    """Достаёт имя модели из step.metadata. Сейчас metadata в большинстве
    шагов пуст (модель туда не пишет LLM-клиент), поэтому почти всегда вернёт "—".
    Когда LLM client'е появится патч — он положит {"model": "..."} в metadata,
    и эта функция сразу его подхватит без изменений view'a.

    Также проверяем legacy-ключи: metadata.model_id, metadata.llm.model.
    """
    meta = content.get("metadata") or {}
    if not isinstance(meta, dict):
        return "—"
    for key in ("model", "model_id", "model_name"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            return v
    llm = meta.get("llm")
    if isinstance(llm, dict):
        v = llm.get("model") or llm.get("name")
        if isinstance(v, str) and v:
            return v
    return "—"


def _quantile(sorted_values: list[float], q: float) -> float:
    """p-квантиль из уже отсортированной asc-последовательности (nearest-rank).
    Возвращает 0.0 для пустого входа. numpy не тащим — масштаб не требует.
    """
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    # nearest-rank, как `numpy.quantile(..., method='lower')` для простоты.
    idx = max(0, min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _parse_uuid_or_none(raw: str | None) -> UUID | None:
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        raise HTTPException(400, f"некорректный UUID: {raw!r}")


def _as_dt(value: Any) -> datetime | None:
    """LanceDB возвращает timestamp как naive datetime; нормализуем к aware UTC
    для сравнений с now(UTC)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return None


@router.get("", response_class=HTMLResponse)
def list_llm_calls(
    request: Request,
    step_type: str = "all",
    trajectory_id: str | None = None,
    goal_id: str | None = None,
    since_minutes: int = 0,
    min_tokens: int = 0,
    limit: int = 200,
    episodic: EpisodicStore = Depends(get_episodic),
) -> HTMLResponse:
    """HTML-страница со списком LLM-вызовов + фильтры + сводные метрики."""
    # --- normalize / validate ---
    if step_type not in _ALL_FILTER_OPTIONS:
        raise HTTPException(
            400, f"step_type must be one of: {', '.join(_ALL_FILTER_OPTIONS)}"
        )
    traj_uuid = _parse_uuid_or_none(trajectory_id)
    goal_uuid = _parse_uuid_or_none(goal_id)
    since_minutes = max(0, since_minutes)
    min_tokens = max(0, min_tokens)
    limit = max(1, min(limit, 1000))

    # LanceDB filter DSL ограничен — recent_steps вернёт большой батч, фильтруем
    # в Python. Берём с запасом ×5, чтобы после фильтра ещё осталось limit'ов.
    raw_rows = episodic.recent_steps(limit=limit * 5)

    cutoff: datetime | None = None
    if since_minutes > 0:
        cutoff = datetime.now(UTC) - timedelta(minutes=since_minutes)

    # --- main filter pass (sequential, easy to debug) ---
    matched: list[dict[str, Any]] = []
    for r in raw_rows:
        st = r.get("step_type")
        if st not in _LLM_STEP_TYPES:
            continue
        if step_type != "all" and st != step_type:
            continue
        if traj_uuid is not None and r.get("trajectory_id") != str(traj_uuid):
            continue
        if goal_uuid is not None and r.get("goal_id") != str(goal_uuid):
            continue
        tokens = int(r.get("cost_tokens") or 0)
        if tokens < min_tokens:
            continue
        ts = _as_dt(r.get("timestamp"))
        if cutoff is not None and (ts is None or ts < cutoff):
            continue

        try:
            content = jsonlib.loads(r.get("content_json") or "{}")
        except jsonlib.JSONDecodeError:
            content = {}

        matched.append(
            {
                "id": r.get("id"),
                "trajectory_id": r.get("trajectory_id"),
                "goal_id": r.get("goal_id"),
                "step_type": st,
                "timestamp": r.get("timestamp"),
                "cost_tokens": tokens,
                "cost_latency": float(r.get("cost_latency") or 0.0),
                "model": _extract_model(content),
            }
        )

    # Sort by timestamp desc (raw_rows уже отсортирован — пересортируем для
    # надёжности, т.к. recent_steps мог отдать частично-отсортированное окно).
    matched.sort(key=lambda x: x["timestamp"] or datetime.min, reverse=True)
    matched = matched[:limit]

    # --- aggregates ---
    total_calls = len(matched)
    total_tokens = sum(r["cost_tokens"] for r in matched)
    latencies = sorted(r["cost_latency"] for r in matched)
    mean_latency = (sum(latencies) / total_calls) if total_calls else 0.0
    p50_latency = _quantile(latencies, 0.5)
    p95_latency = _quantile(latencies, 0.95)
    mean_tokens = (total_tokens / total_calls) if total_calls else 0.0

    # Breakdown by step_type (только по типам, которые попали в matched).
    breakdown: dict[str, dict[str, int]] = {}
    for st in _LLM_STEP_TYPES:
        breakdown[st] = {"calls": 0, "tokens": 0}
    for r in matched:
        st = r["step_type"]
        breakdown[st]["calls"] += 1
        breakdown[st]["tokens"] += r["cost_tokens"]
    max_calls = max((b["calls"] for b in breakdown.values()), default=0)
    max_tokens_b = max((b["tokens"] for b in breakdown.values()), default=0)

    return templates.TemplateResponse(
        request,
        "llm/list.html",
        {
            # filters (для URL state + form repopulate)
            "selected_step_type": step_type,
            "selected_trajectory_id": trajectory_id or "",
            "selected_goal_id": goal_id or "",
            "selected_since_minutes": since_minutes,
            "selected_min_tokens": min_tokens,
            "selected_limit": limit,
            "all_step_types": _ALL_FILTER_OPTIONS,
            # rows + stats
            "rows": matched,
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "mean_latency": mean_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "mean_tokens": mean_tokens,
            "breakdown": breakdown,
            "max_breakdown_calls": max_calls,
            "max_breakdown_tokens": max_tokens_b,
        },
    )
