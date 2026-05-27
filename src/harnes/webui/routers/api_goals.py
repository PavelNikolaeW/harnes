"""Goals: список, дерево, approve/reject, создание."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
    StateChangePredicate,
    StructuralPredicate,
)
from harnes.goals.store import GoalRepository
from harnes.webui.config import WebuiSettings, get_webui_settings
from harnes.webui.deps import get_goal_repo
from harnes.webui.templating import templates

router = APIRouter()


def _all_goals(repo: GoalRepository) -> list[Goal]:
    out: list[Goal] = []
    for s in GoalStatus:
        out.extend(repo.list_by_status(s))
    out.sort(key=lambda g: g.updated_at, reverse=True)
    return out


@router.get("", response_class=HTMLResponse)
def list_goals(
    request: Request,
    status: str | None = None,
    goal_class: str | None = None,
    origin: str | None = None,
    repo: GoalRepository = Depends(get_goal_repo),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> HTMLResponse:
    """Все цели с фильтрами (status / class / origin)."""
    if status:
        try:
            goals = repo.list_by_status(GoalStatus(status))
        except ValueError:
            raise HTTPException(400, f"unknown status: {status}")
    else:
        goals = _all_goals(repo)

    if goal_class:
        try:
            gc = GoalClass(goal_class)
            goals = [g for g in goals if g.goal_class == gc]
        except ValueError:
            raise HTTPException(400, f"unknown class: {goal_class}")

    if origin:
        try:
            org = Origin(origin)
            goals = [g for g in goals if g.origin == org]
        except ValueError:
            raise HTTPException(400, f"unknown origin: {origin}")

    return templates.TemplateResponse(
        request,
        "goals/list.html",
        {
            "goals": goals,
            "selected_status": status,
            "selected_class": goal_class,
            "selected_origin": origin,
            "all_statuses": [s.value for s in GoalStatus],
            "all_classes": [c.value for c in GoalClass],
            "all_origins": [o.value for o in Origin],
            "read_only": cfg.read_only,
        },
    )


@router.get("/new", response_class=HTMLResponse)
def new_goal_form(
    request: Request,
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> HTMLResponse:
    if cfg.read_only:
        raise HTTPException(403, "read-only mode")
    return templates.TemplateResponse(
        request,
        "goals/new.html",
        {
            "all_classes": [c.value for c in GoalClass],
            "predicate_kinds": ["judge", "structural", "state_change"],
        },
    )


@router.post("", response_class=HTMLResponse)
def create_goal(
    request: Request,
    description: str = Form(...),
    goal_class: str = Form("task"),
    priority: int = Form(0),
    predicate_kind: str = Form("judge"),
    criterion: str = Form(""),
    repo: GoalRepository = Depends(get_goal_repo),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    if cfg.read_only:
        raise HTTPException(403, "read-only mode")

    if predicate_kind == "judge":
        predicate = JudgePredicate(criterion=criterion or "operator-judged complete")
    elif predicate_kind == "structural":
        predicate = StructuralPredicate(expected_schema={"type": "object"})
    elif predicate_kind == "state_change":
        predicate = StateChangePredicate(check_tool_id="", expected_outcome={})
    else:
        raise HTTPException(400, f"unknown predicate_kind: {predicate_kind}")

    goal = Goal(
        description=description.strip(),
        goal_class=GoalClass(goal_class),
        priority=priority,
        predicate_of_success=predicate,
        origin=Origin.OPERATOR,
        originator="webui",
    )
    repo.create(goal)
    return RedirectResponse(url=f"/goals/{goal.id}", status_code=303)


@router.get("/{goal_id}", response_class=HTMLResponse)
def goal_detail(
    request: Request,
    goal_id: str,
    repo: GoalRepository = Depends(get_goal_repo),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> HTMLResponse:
    """Детальная карточка цели: poll, predicate, бюджет, дерево потомков."""
    try:
        gid = UUID(goal_id)
    except ValueError:
        raise HTTPException(404, "invalid goal id")

    goal = repo.get(gid)
    if goal is None:
        raise HTTPException(404, f"goal {goal_id} not found")

    parent = repo.get(goal.parent_id) if goal.parent_id else None

    # Тre дерево детей (рекурсивно). Для исследовательской консоли — без лимита.
    def _subtree(node: Goal, depth: int = 0) -> list[dict]:
        out = [{"goal": node, "depth": depth}]
        for child in repo.list_children(node.id):
            out.extend(_subtree(child, depth + 1))
        return out

    tree = _subtree(goal)
    deps = [repo.get(d) for d in goal.depends_on]
    deps = [d for d in deps if d is not None]

    return templates.TemplateResponse(
        request,
        "goals/detail.html",
        {
            "goal": goal,
            "parent": parent,
            "tree": tree,
            "depends_on": deps,
            "read_only": cfg.read_only,
        },
    )


@router.post("/{goal_id}/approve", response_class=HTMLResponse)
def approve(
    request: Request,
    goal_id: str,
    repo: GoalRepository = Depends(get_goal_repo),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    if cfg.read_only:
        raise HTTPException(403, "read-only mode")
    try:
        gid = UUID(goal_id)
    except ValueError:
        raise HTTPException(404, "invalid goal id")
    try:
        repo.approve(gid)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return RedirectResponse(url=f"/goals/{goal_id}", status_code=303)


@router.post("/{goal_id}/reject", response_class=HTMLResponse)
def reject(
    request: Request,
    goal_id: str,
    reason: str = Form("rejected via webui"),
    repo: GoalRepository = Depends(get_goal_repo),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    if cfg.read_only:
        raise HTTPException(403, "read-only mode")
    try:
        gid = UUID(goal_id)
    except ValueError:
        raise HTTPException(404, "invalid goal id")
    try:
        repo.reject(gid, reason)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return RedirectResponse(url=f"/goals/{goal_id}", status_code=303)


@router.post("/{goal_id}/abandon", response_class=HTMLResponse)
def abandon(
    request: Request,
    goal_id: str,
    reason: str = Form("abandoned via webui"),
    repo: GoalRepository = Depends(get_goal_repo),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    """Ручная остановка active/pending цели — переводит в ABANDONED."""
    if cfg.read_only:
        raise HTTPException(403, "read-only mode")
    try:
        gid = UUID(goal_id)
    except ValueError:
        raise HTTPException(404, "invalid goal id")
    goal = repo.get(gid)
    if goal is None:
        raise HTTPException(404, f"goal {goal_id} not found")
    if goal.status in (GoalStatus.DONE, GoalStatus.FAILED, GoalStatus.ABANDONED):
        raise HTTPException(400, f"goal is in terminal status {goal.status.value}")
    goal.status = GoalStatus.ABANDONED
    goal.metadata = {**goal.metadata, "abandon_reason": reason}
    repo.update(goal)
    return RedirectResponse(url=f"/goals/{goal_id}", status_code=303)


@router.post("/bulk", response_class=HTMLResponse)
def bulk_action(
    request: Request,
    action: str = Form(...),
    ids: str = Form(...),
    reason: str = Form("bulk action via webui"),
    repo: GoalRepository = Depends(get_goal_repo),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    """Apply action to many goals at once. `ids` — CSV of UUID strings.

    Действия:
    - approve  — PENDING_APPROVAL → PENDING
    - reject   — PENDING_APPROVAL → ABANDONED
    - abandon  — любой не-терминальный → ABANDONED
    Per-goal ошибки агрегируются (не валим всё на одной).
    """
    if cfg.read_only:
        raise HTTPException(403, "read-only mode")
    if action not in ("approve", "reject", "abandon"):
        raise HTTPException(400, f"unknown action: {action}")

    id_list: list[UUID] = []
    for raw in ids.split(","):
        s = raw.strip()
        if not s:
            continue
        try:
            id_list.append(UUID(s))
        except ValueError:
            raise HTTPException(400, f"invalid uuid in ids: {s!r}")
    if not id_list:
        raise HTTPException(400, "ids cannot be empty")

    applied = 0
    errors: list[str] = []
    for gid in id_list:
        try:
            if action == "approve":
                repo.approve(gid)
            elif action == "reject":
                repo.reject(gid, reason)
            else:  # abandon
                goal = repo.get(gid)
                if goal is None:
                    errors.append(f"{gid}: not found")
                    continue
                if goal.status in (GoalStatus.DONE, GoalStatus.FAILED, GoalStatus.ABANDONED):
                    errors.append(f"{gid}: terminal status {goal.status.value}")
                    continue
                goal.status = GoalStatus.ABANDONED
                goal.metadata = {**goal.metadata, "abandon_reason": reason}
                repo.update(goal)
            applied += 1
        except (KeyError, ValueError) as exc:
            errors.append(f"{gid}: {exc}")

    # Redirect on referer (preserves filter context).
    ref = request.headers.get("referer", "/goals")
    sep = "&" if "?" in ref else "?"
    flash = f"bulk_action={action}&bulk_applied={applied}"
    if errors:
        flash += f"&bulk_errors={len(errors)}"
    return RedirectResponse(url=f"{ref}{sep}{flash}", status_code=303)
