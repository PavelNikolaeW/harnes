"""Skills view — список бандлов + per-version метрики.

Read-only: изменения скиллов остаются за reflect и CLI. См. § 9 архитектуры.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select

from harnes.skills.schema import Skill, SkillMetrics, SkillStatus
from harnes.skills.store import InvocationRow, SkillRegistry
from harnes.webui.deps import get_skill_registry
from harnes.webui.templating import templates

router = APIRouter()


def _list_versions(registry: SkillRegistry, skill_id: str) -> list[dict[str, Any]]:
    """Уникальные версии скилла из invocations + агрегированные метрики per-version.

    SkillRegistry хранит invocations плоско; собираем DISTINCT version → metrics.
    """
    with Session(registry.engine) as s:
        versions = list(
            s.exec(
                select(InvocationRow.skill_version)
                .where(InvocationRow.skill_id == skill_id)
                .distinct()
            ).all()
        )
        latest_ts: dict[str, datetime] = {}
        for v in versions:
            row = s.exec(
                select(InvocationRow.timestamp)
                .where(InvocationRow.skill_id == skill_id)
                .where(InvocationRow.skill_version == v)
                .order_by(InvocationRow.timestamp.desc())
                .limit(1)
            ).first()
            if row is not None:
                latest_ts[v] = row

    out = []
    for v in versions:
        metrics = registry.get_metrics(skill_id, version=v)
        out.append(
            {
                "version": v,
                "metrics": metrics,
                "last_invocation": latest_ts.get(v),
            }
        )
    out.sort(key=lambda r: (r["last_invocation"] or datetime.min), reverse=True)
    return out


@router.get("", response_class=HTMLResponse)
def list_skills(
    request: Request,
    status: str | None = None,
    registry: SkillRegistry = Depends(get_skill_registry),
) -> HTMLResponse:
    """Все бандлы из bundles_dir + агрегированные метрики."""
    if status:
        try:
            SkillStatus(status)
        except ValueError:
            raise HTTPException(400, f"unknown status: {status}")

    skills: list[Skill] = registry.load_all()
    if status:
        skills = [s for s in skills if s.status.value == status]

    rows = []
    for skill in skills:
        # Aggregated по всем версиям — даёт общий success rate.
        metrics = registry.get_metrics(skill.id)
        rows.append({"skill": skill, "metrics": metrics})

    rows.sort(key=lambda r: (
        0 if r["skill"].status == SkillStatus.ACTIVE else
        1 if r["skill"].status == SkillStatus.EXPERIMENTAL else 2,
        r["skill"].name,
    ))

    return templates.TemplateResponse(
        request,
        "skills/list.html",
        {
            "rows": rows,
            "selected_status": status,
            "all_statuses": [s.value for s in SkillStatus],
        },
    )


@router.get("/{skill_id}", response_class=HTMLResponse)
def skill_detail(
    request: Request,
    skill_id: str,
    registry: SkillRegistry = Depends(get_skill_registry),
) -> HTMLResponse:
    skill = registry.get(skill_id)
    if skill is None:
        raise HTTPException(404, f"skill {skill_id} not found")

    versions = _list_versions(registry, skill_id)
    aggregated: SkillMetrics = registry.get_metrics(skill_id)

    return templates.TemplateResponse(
        request,
        "skills/detail.html",
        {
            "skill": skill,
            "versions": versions,
            "aggregated": aggregated,
        },
    )
