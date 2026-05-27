"""Jinja2 templates + shared template-helpers.

Singleton Jinja2Templates на пакет — все routers импортируют отсюда.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------- Filters ----------


def pretty_json(value: Any) -> str:
    """Отформатированный JSON для вывода в <pre>. None-safe."""
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


_RE_RU_NS = re.compile(r"\.\d+\+\d\d:\d\d$")


def short_dt(dt: datetime | str | None) -> str:
    """`2026-05-27 14:23:01` без таймзоны и микросекунд."""
    if dt is None:
        return ""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def relative_time(dt: datetime | str | None) -> str:
    """Человекочитаемое смещение: '5s ago', '12m ago', '3h ago', '2d ago'."""
    if dt is None:
        return "—"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return dt
    now = datetime.now(UTC) if dt.tzinfo else datetime.now()
    seconds = (now - dt).total_seconds()
    if seconds < 0:
        return short_dt(dt)
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)}d ago"
    return short_dt(dt)


def short_uuid(u: Any) -> str:
    """Первые 8 hex'ов UUID — для tightly-packed таблиц."""
    if u is None:
        return ""
    s = str(u)
    if len(s) >= 8:
        return s[:8]
    return s


def status_class(status: str | None) -> str:
    """Tailwind classes для badge статуса (goal/trajectory/verify)."""
    if not status:
        return "bg-stone-100 text-stone-700"
    s = str(status).lower()
    if s in ("done", "success", "ok", "active"):
        return "bg-emerald-50 text-emerald-800 border-emerald-200"
    if s in ("failed", "failure", "fail", "error", "rejected"):
        return "bg-red-50 text-red-800 border-red-200"
    if s in ("pending", "pending_approval", "experimental", "warning"):
        return "bg-amber-50 text-amber-800 border-amber-200"
    if s in ("abandoned", "deprecated", "suspended", "budget_exceeded"):
        return "bg-stone-100 text-stone-600 border-stone-300"
    if s in ("partial", "undetermined"):
        return "bg-sky-50 text-sky-800 border-sky-200"
    return "bg-stone-100 text-stone-700 border-stone-200"


def step_color(step_type: str | None) -> str:
    """Цветовой класс для левой полосы trajectory step."""
    s = (step_type or "").lower()
    return {
        "thought": "border-sky-400 bg-sky-50/40",
        "plan": "border-violet-400 bg-violet-50/40",
        "action": "border-amber-400 bg-amber-50/40",
        "observation": "border-emerald-400 bg-emerald-50/40",
        "critique": "border-rose-400 bg-rose-50/40",
        "retry_note": "border-stone-400 bg-stone-100/60",
    }.get(s, "border-stone-300 bg-stone-50/40")


templates.env.filters["pretty_json"] = pretty_json
templates.env.filters["short_dt"] = short_dt
templates.env.filters["relative_time"] = relative_time
templates.env.filters["short_uuid"] = short_uuid
templates.env.filters["status_class"] = status_class
templates.env.filters["step_color"] = step_color
