"""Memory volumes viewer — observability на объёмы 4 слоёв памяти.

См. § 13 архитектуры: episodic (LanceDB) / semantic (Qdrant) / world (Neo4j+Graphiti)
/ procedural (git skills + SQLite invocations). Consolidation/decay — open question,
здесь только current snapshot и тренд episodic за last 7d.

Mount под `prefix="/memory/volumes"` (отдельный router, чтобы не конфликтовать
с `/memory` от api_memory).
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from harnes.config import Settings
from harnes.memory.episodic import EpisodicStore
from harnes.memory.semantic import SemanticStore
from harnes.memory.world import WorldModelStore
from harnes.skills.store import InvocationRow, SkillRegistry
from harnes.webui.deps import (
    get_agent_settings,
    get_episodic,
    get_semantic,
    get_skill_registry,
    get_world,
)
from harnes.webui.templating import templates

log = structlog.get_logger()

router = APIRouter()


# Defensive limit для episodic дампа — LanceDB recent_* возвращает list, считаем
# через len(). Для >100k records это станет тормозом — TODO: добавить native
# count() в EpisodicStore (LanceDB поддерживает count_rows() на table'е).
_EPISODIC_DUMP_LIMIT = 100_000
_GROWTH_DAYS = 7


def _as_aware_utc(value: Any) -> datetime | None:
    """LanceDB отдаёт naive datetime — нормализуем к aware UTC. None-safe."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def _episodic_snapshot(episodic: EpisodicStore) -> dict[str, Any]:
    """Counts + 7-day growth bars + top-5 goals by step count.

    Возвращает dict; на ошибке — {"error": str}.
    """
    try:
        trajectories = episodic.recent_trajectories(limit=_EPISODIC_DUMP_LIMIT)
        steps = episodic.recent_steps(limit=_EPISODIC_DUMP_LIMIT)
    except Exception as exc:
        log.warning("memory.volumes.episodic.failed", error=str(exc))
        return {"error": f"episodic dump failed: {exc}"}

    # 7-day growth: bucketize по date(ts), считаем steps на day.
    # Гарантируем 7 слотов даже при пустых днях (zero-bars).
    now = datetime.now(UTC)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    horizon = [today - timedelta(days=i) for i in range(_GROWTH_DAYS - 1, -1, -1)]
    buckets: dict[str, int] = {d.strftime("%Y-%m-%d"): 0 for d in horizon}
    cutoff = horizon[0]

    for s in steps:
        ts = _as_aware_utc(s.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        key = ts.replace(hour=0, minute=0, second=0, microsecond=0).strftime(
            "%Y-%m-%d"
        )
        if key in buckets:
            buckets[key] += 1

    growth_rows = [
        {"label": k[5:], "date": k, "steps": v} for k, v in buckets.items()
    ]
    max_growth = max((r["steps"] for r in growth_rows), default=0)

    # Top-5 goals by step count.
    goal_counts: Counter[str] = Counter()
    for s in steps:
        gid = s.get("goal_id") or ""
        if gid:
            goal_counts[gid] += 1
    top_goals = [
        {"goal_id": gid, "steps": cnt}
        for gid, cnt in goal_counts.most_common(5)
    ]

    # Дополнительно — самая ранняя/поздняя timestamp для контекста.
    ts_values = [_as_aware_utc(s.get("timestamp")) for s in steps]
    ts_values = [t for t in ts_values if t is not None]
    earliest = min(ts_values) if ts_values else None
    latest = max(ts_values) if ts_values else None

    return {
        "trajectories": len(trajectories),
        "steps": len(steps),
        "capped": len(steps) >= _EPISODIC_DUMP_LIMIT,
        "growth_rows": growth_rows,
        "max_growth": max_growth,
        "top_goals": top_goals,
        "earliest": earliest,
        "latest": latest,
    }


def _semantic_snapshot(semantic: SemanticStore | None) -> dict[str, Any]:
    """Точный count + dim из qdrant. На ошибке — {"error": ...}."""
    if semantic is None:
        return {"available": False}
    try:
        # exact=True гарантирует точный count (не оценку), что нам и нужно
        # для observability — таблица будет небольшая (десятки тысяч).
        count_resp = semantic.client.count(
            collection_name=semantic.collection, exact=True
        )
        total = int(getattr(count_resp, "count", 0))
    except Exception as exc:
        log.warning("memory.volumes.semantic.count_failed", error=str(exc))
        return {"available": True, "error": f"qdrant count failed: {exc}"}

    dim: int | None = None
    distance: str | None = None
    try:
        info = semantic.client.get_collection(collection_name=semantic.collection)
        # Структура: info.config.params.vectors — VectorParams ИЛИ dict (named).
        vectors = info.config.params.vectors  # type: ignore[attr-defined]
        if hasattr(vectors, "size"):
            dim = int(vectors.size)
            distance = str(getattr(vectors, "distance", "") or "")
        elif isinstance(vectors, dict) and vectors:
            # Названные векторы — берём первый.
            first = next(iter(vectors.values()))
            dim = int(getattr(first, "size", 0)) or None
            distance = str(getattr(first, "distance", "") or "")
    except Exception as exc:
        log.debug("memory.volumes.semantic.get_collection_failed", error=str(exc))
        # count уже получен — просто без dim.

    return {
        "available": True,
        "total": total,
        "dim": dim,
        "distance": distance,
        "collection": semantic.collection,
    }


def _world_snapshot(
    world: WorldModelStore | None, settings: Settings
) -> dict[str, Any]:
    """Neo4j counts + breakdown by label. None world → {"available": False}.

    Neo4j driver открывается локально (короткоживущая сессия), потом close().
    Любая ошибка — заворачиваем в {"available": True, "error": ...} чтобы
    отрисовать "unreachable", а не ронять страницу.
    """
    if world is None:
        return {"available": False}

    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return {"available": True, "error": f"neo4j driver import failed: {exc}"}

    try:
        driver = GraphDatabase.driver(
            settings.memory.neo4j_uri,
            auth=(settings.memory.neo4j_user, settings.memory.neo4j_password),
        )
    except Exception as exc:
        return {"available": True, "error": f"neo4j driver init failed: {exc}"}

    try:
        with driver.session() as s:
            nodes = s.run("MATCH (n) RETURN count(n) AS c").single()
            node_count = int(nodes["c"]) if nodes else 0

            edges = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
            edge_count = int(edges["c"]) if edges else 0

            label_rows: list[dict[str, Any]] = []
            # labels(n) может быть пустым массивом для безлейбловых узлов.
            for rec in s.run(
                "MATCH (n) UNWIND labels(n) AS lbl "
                "RETURN lbl, count(*) AS c ORDER BY c DESC"
            ):
                label_rows.append({"label": str(rec["lbl"]), "count": int(rec["c"])})
    except Exception as exc:
        try:
            driver.close()
        except Exception:
            pass
        return {"available": True, "error": f"neo4j query failed: {exc}"}

    try:
        driver.close()
    except Exception:
        pass

    return {
        "available": True,
        "nodes": node_count,
        "edges": edge_count,
        "labels": label_rows,
    }


def _procedural_snapshot(skill_registry: SkillRegistry) -> dict[str, Any]:
    """Skill counts (by status) + total invocations recorded."""
    try:
        skills = skill_registry.load_all()
    except Exception as exc:
        return {"error": f"skill load failed: {exc}"}

    by_status: Counter[str] = Counter()
    for sk in skills:
        # status — Enum (str-derived); .value безопасно.
        st = getattr(sk.status, "value", str(sk.status))
        by_status[st] += 1

    # Invocations — SQLModel count. Прямой select(count) чтобы не материализовать
    # все строки. Engine у registry — public, см. skills.store.SkillRegistry.
    invocations: int | None = None
    invocations_error: str | None = None
    try:
        from sqlalchemy import func
        from sqlmodel import Session, select

        with Session(skill_registry.engine) as s:
            invocations = int(
                s.exec(select(func.count(InvocationRow.id))).one()  # type: ignore[arg-type]
            )
    except Exception as exc:
        log.warning("memory.volumes.procedural.invocations_failed", error=str(exc))
        invocations_error = f"invocations query failed: {exc}"

    return {
        "total_skills": len(skills),
        "by_status": dict(by_status),
        "invocations": invocations,
        "invocations_error": invocations_error,
    }


@router.get("", response_class=HTMLResponse)
def memory_volumes(
    request: Request,
    episodic: EpisodicStore = Depends(get_episodic),
    semantic: SemanticStore | None = Depends(get_semantic),
    world: WorldModelStore | None = Depends(get_world),
    skill_registry: SkillRegistry = Depends(get_skill_registry),
    settings: Settings = Depends(get_agent_settings),
) -> HTMLResponse:
    """4-layer memory volumes snapshot + episodic 7-day growth.

    Каждый слой обрабатывается изолированно — отказ одного backend'а не валит
    страницу. См. § 13 архитектуры.
    """
    ep = _episodic_snapshot(episodic)
    sem = _semantic_snapshot(semantic)
    wrld = _world_snapshot(world, settings)
    proc = _procedural_snapshot(skill_registry)

    return templates.TemplateResponse(
        request,
        "memory/volumes.html",
        {
            "episodic": ep,
            "semantic": sem,
            "world": wrld,
            "procedural": proc,
            "growth_days": _GROWTH_DAYS,
            "episodic_dump_limit": _EPISODIC_DUMP_LIMIT,
            "now": datetime.now(UTC),
        },
    )
