"""Trajectory Replay — пошаговый rewind/forward inspector.

Альтернативный view на trajectory: вместо всей timeline показываем ОДИН шаг
с возможностью двигаться вперёд/назад по клавишам/кнопкам. Research-фича:
позволяет посмотреть ровно тот контекст и ровно те данные, которые видел
агент в каждый момент решения.

Регистрируется через `app.include_router(router, prefix="/replay")`.
"""
from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.goals.store import GoalRepository
from harnes.memory.episodic import EpisodicStore
from harnes.webui.deps import get_episodic, get_goal_repo
from harnes.webui.templating import templates

router = APIRouter()


def _parse_content(content_json: str) -> dict:
    """LanceDB хранит content как JSON-строку. None-safe."""
    if not content_json:
        return {}
    try:
        return json.loads(content_json)
    except json.JSONDecodeError:
        return {"_raw": content_json}


@router.get("/{trajectory_id}", response_class=HTMLResponse)
def replay_trajectory(
    request: Request,
    trajectory_id: str,
    episodic: EpisodicStore = Depends(get_episodic),
    goal_repo: GoalRepository = Depends(get_goal_repo),
) -> HTMLResponse:
    """Step-by-step replay одной trajectory.

    Возвращает HTML страницу с Alpine.js-управляемым inspector'ом:
    клавиши ←/→ и j/k переключают шаг, ?step=N в URL восстанавливает
    позицию между перезагрузками.
    """
    try:
        tid = UUID(trajectory_id)
    except ValueError:
        raise HTTPException(404, "invalid trajectory id")

    meta = episodic.get_trajectory_meta(tid)
    if meta is None:
        raise HTTPException(404, f"trajectory {trajectory_id} not found")

    steps_raw = episodic.get_steps(tid)
    steps = []
    cumulative_tokens_at: list[int] = []
    cumulative_latency_at: list[float] = []
    running_tokens = 0
    running_latency = 0.0
    for s in steps_raw:
        running_tokens += int(s.get("cost_tokens") or 0)
        running_latency += float(s.get("cost_latency") or 0.0)
        cumulative_tokens_at.append(running_tokens)
        cumulative_latency_at.append(running_latency)
        steps.append(
            {
                **s,
                "content": _parse_content(s.get("content_json", "")),
            }
        )

    goal = None
    goal_id_str = meta.get("goal_id")
    if goal_id_str:
        try:
            goal = goal_repo.get(UUID(goal_id_str))
        except (ValueError, Exception):
            goal = None

    return templates.TemplateResponse(
        request,
        "replay/trajectory.html",
        {
            "meta": meta,
            "steps": steps,
            "goal": goal,
            "cumulative_tokens_at": cumulative_tokens_at,
            "cumulative_latency_at": cumulative_latency_at,
        },
    )
