"""Trajectories — список + детальный inspector."""
from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.goals.store import GoalRepository
from harnes.memory.episodic import EpisodicStore
from harnes.react.schema import TrajectoryStatus
from harnes.webui.deps import get_episodic, get_goal_repo
from harnes.webui.templating import templates

router = APIRouter()


@router.get("", response_class=HTMLResponse)
def list_trajectories(
    request: Request,
    status: str | None = None,
    goal_id: str | None = None,
    limit: int = 50,
    episodic: EpisodicStore = Depends(get_episodic),
    goal_repo: GoalRepository = Depends(get_goal_repo),
) -> HTMLResponse:
    """Recent N трейекторий — meta-таблица. С опциональными status/goal_id фильтрами."""
    limit = max(1, min(limit, 500))
    if status:
        try:
            TrajectoryStatus(status)
        except ValueError:
            raise HTTPException(400, f"unknown status: {status}")

    rows: list = []
    goal_obj = None
    if goal_id:
        try:
            gid = UUID(goal_id)
        except ValueError:
            raise HTTPException(400, f"invalid goal_id: {goal_id}")
        rows = episodic.list_trajectories_for_goal(gid)
        rows.sort(key=lambda r: r.get("started_at") or "", reverse=True)
        if status:
            rows = [r for r in rows if r.get("status") == status]
        rows = rows[:limit]
        goal_obj = goal_repo.get(gid)
    else:
        rows = episodic.recent_trajectories(limit=limit, status=status)

    return templates.TemplateResponse(
        request,
        "trajectories/list.html",
        {
            "trajectories": rows,
            "selected_status": status,
            "selected_goal_id": goal_id,
            "selected_goal": goal_obj,
            "all_statuses": [s.value for s in TrajectoryStatus],
            "limit": limit,
            "diff_enabled": len(rows) >= 2,
        },
    )


def _parse_content(content_json: str) -> dict:
    """LanceDB хранит content как JSON-строку. None-safe."""
    if not content_json:
        return {}
    try:
        return json.loads(content_json)
    except json.JSONDecodeError:
        return {"_raw": content_json}


@router.get("/{trajectory_id}", response_class=HTMLResponse)
def trajectory_detail(
    request: Request,
    trajectory_id: str,
    episodic: EpisodicStore = Depends(get_episodic),
    goal_repo: GoalRepository = Depends(get_goal_repo),
) -> HTMLResponse:
    """Полная trajectory: meta + timeline шагов."""
    try:
        tid = UUID(trajectory_id)
    except ValueError:
        raise HTTPException(404, "invalid trajectory id")

    meta = episodic.get_trajectory_meta(tid)
    if meta is None:
        raise HTTPException(404, f"trajectory {trajectory_id} not found")

    steps_raw = episodic.get_steps(tid)
    steps = []
    for s in steps_raw:
        steps.append(
            {
                **s,
                "content": _parse_content(s.get("content_json", "")),
            }
        )

    # Меty-разбивка: суммы cost'ов и кол-во шагов каждого типа.
    type_counts: dict[str, int] = {}
    total_tokens = 0
    total_latency = 0.0
    for s in steps_raw:
        type_counts[s["step_type"]] = type_counts.get(s["step_type"], 0) + 1
        total_tokens += int(s.get("cost_tokens") or 0)
        total_latency += float(s.get("cost_latency") or 0.0)

    goal = None
    goal_id_str = meta.get("goal_id")
    if goal_id_str:
        try:
            goal = goal_repo.get(UUID(goal_id_str))
        except (ValueError, Exception):
            goal = None

    # Final state — может быть JSON или произвольный текст.
    final_state_str = meta.get("final_state_json") or ""
    final_state: object = final_state_str
    if final_state_str:
        try:
            final_state = json.loads(final_state_str)
        except json.JSONDecodeError:
            pass

    return templates.TemplateResponse(
        request,
        "trajectories/detail.html",
        {
            "meta": meta,
            "steps": steps,
            "type_counts": type_counts,
            "total_tokens": total_tokens,
            "total_latency": total_latency,
            "goal": goal,
            "final_state": final_state,
        },
    )
