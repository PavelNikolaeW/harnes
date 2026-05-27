"""Skills view — список бандлов + per-version метрики + минимальный edit.

Edit-mode (POST /{id}) переписывает prompt_template / status / description
прямо в YAML-бандле; reflect-pipeline отдельно. См. § 9 архитектуры.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError
from sqlmodel import Session, select

from harnes.skills.schema import Skill, SkillMetrics, SkillOrigin, SkillStatus
from harnes.skills.store import InvocationRow, SkillRegistry
from harnes.webui.config import WebuiSettings, get_webui_settings
from harnes.webui.deps import get_skill_registry
from harnes.webui.templating import templates

router = APIRouter()


def _bump_patch(version: str) -> str:
    """Increment последний сегмент semver-строки на 1.

    "0.0.1" -> "0.0.2", "1.4" -> "1.5", "1.2.3.4" -> "1.2.3.5".
    Если последний сегмент float — оставляем dot ("1.2.3.5.0" парсится
    через int(float()), но fallback ниже). Если последний сегмент не
    парсится как число — добавляем ".1".
    """
    if not version:
        return "0.0.1"
    parts = version.split(".")
    last = parts[-1]
    try:
        bumped = str(int(last) + 1)
        return ".".join([*parts[:-1], bumped])
    except ValueError:
        # Не int — пробуем float (например "0.0.1a" не пройдёт; для
        # "1.2" последний "2" уже int — этот ветка только для нечислового).
        try:
            bumped_f = float(last) + 1.0
            return ".".join([*parts[:-1], str(bumped_f)])
        except ValueError:
            return version + ".1"


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

    cfg = get_webui_settings()
    return templates.TemplateResponse(
        request,
        "skills/detail.html",
        {
            "skill": skill,
            "versions": versions,
            "aggregated": aggregated,
            "read_only": cfg.read_only,
            "bumped_version": _bump_patch(skill.version),
        },
    )


@router.post("/{skill_id}", response_class=HTMLResponse)
def edit_skill(
    request: Request,
    skill_id: str,
    prompt_template: str = Form(...),
    status: str = Form(...),
    description: str = Form(...),
    # FastAPI Form bool парсит "on"/"true"/"1" → True, "off"/"false"/"0" → False.
    # HTML-checkbox при unchecked НЕ шлёт поле вовсе, поэтому даём дефолт False.
    bump_version: bool = Form(False),
    registry: SkillRegistry = Depends(get_skill_registry),
    cfg: WebuiSettings = Depends(get_webui_settings),
) -> RedirectResponse:
    """Operator-edit minimal-set полей бандла: prompt / status / description.

    Если bump_version=true — increment patch, set parent_version_id на
    старую версию, новая статус = ACTIVE. TODO: git-based versioning ожидает
    git_auto_commit; иначе старая версия теряется (мы пишем поверх того же
    YAML-файла).
    """
    if cfg.read_only:
        raise HTTPException(403, "read-only mode")

    skill = registry.get(skill_id)
    if skill is None:
        raise HTTPException(404, f"skill {skill_id} not found")

    try:
        new_status = SkillStatus(status)
    except ValueError:
        raise HTTPException(400, f"unknown status: {status!r}")

    old_version = skill.version
    skill.prompt_template = prompt_template
    skill.description = description.strip()
    skill.updated_at = datetime.now(UTC)
    skill.origin = SkillOrigin.OPERATOR

    if bump_version:
        skill.version = _bump_patch(old_version)
        skill.parent_version_id = old_version
        skill.status = SkillStatus.ACTIVE
    else:
        skill.status = new_status

    try:
        # Re-validate через pydantic — ловим случайные нарушения схемы.
        Skill.model_validate(skill.model_dump(mode="json"))
    except ValidationError as exc:
        raise HTTPException(400, f"skill validation failed: {exc}")

    registry.save(skill)
    return RedirectResponse(url=f"/skills/{skill_id}", status_code=303)
