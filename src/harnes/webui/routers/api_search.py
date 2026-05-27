"""Universal search — single query box across all stores.

Закрывает дыру когда оператор не помнит "куда я положил эту цель / в каком
trajectory он сделал X". Search по goals + trajectories (episodic) + journal
events + skills + eval-runs одним запросом, UI группирует по типу.

Архитектура: server-side фильтр поверх re-fetch'а row'ов из каждого store.
Не самая эффективная архитектура (для prod нужен FTS index), но для dev-объёмов
норм + не требует новой инфраструктуры.

Tolerance: каждый source-query завёрнут в try/except. На сбое (LanceDB closed,
SQLite locked, network failure) — `available=False, error=msg`, остальные
источники продолжают работать. Если store вообще не подключён в `app.state`
(например, Qdrant down при старте) — `available=False, error=None`.
"""
from __future__ import annotations

import json as jsonlib
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from harnes.webui.templating import templates

log = structlog.get_logger()

router = APIRouter()


# ---------- Source identifiers ----------

ALL_SOURCES = ("goals", "trajectories", "journal", "skills", "eval")


def _parse_types(types_csv: str) -> list[str]:
    """Парсим CSV `types`. "all" / пусто → все источники.

    Неизвестные имена тихо игнорируем, чтобы случайный мусор в URL не валил
    страницу.
    """
    if not types_csv or types_csv.strip().lower() == "all":
        return list(ALL_SOURCES)
    requested = {t.strip().lower() for t in types_csv.split(",") if t.strip()}
    return [t for t in ALL_SOURCES if t in requested]


def _try_uuid(q: str) -> UUID | None:
    """Если q — валидный UUID, возвращаем его. Иначе None.

    Используется для exact-match path'ов: goal.get(UUID), trajectory.meta(UUID).
    """
    try:
        return UUID(q.strip())
    except (ValueError, AttributeError):
        return None


# ---------- Per-source search functions ----------


def _search_goals(state: Any, q: str, q_uuid: UUID | None, limit: int) -> dict[str, Any]:
    """Goals: full-text по description (case-insensitive) + UUID exact match.

    Iter всех статусов → flat list → filter. Дороговато при тысячах целей,
    но в v0 их десятки.
    """
    repo = getattr(state, "goal_repo", None)
    if repo is None:
        return {"hits": [], "count": 0, "available": False, "error": None}
    try:
        # UUID-shortcut: один get вместо полного scan'а.
        if q_uuid is not None:
            g = repo.get(q_uuid)
            return {
                "hits": [g] if g is not None else [],
                "count": 1 if g is not None else 0,
                "available": True,
                "error": None,
            }

        from harnes.goals.schema import GoalStatus

        q_lower = q.lower()
        matches = []
        seen_ids: set[UUID] = set()
        for status in GoalStatus:
            for g in repo.list_by_status(status):
                if g.id in seen_ids:
                    continue
                if q_lower in g.description.lower():
                    matches.append(g)
                    seen_ids.add(g.id)
        matches.sort(key=lambda g: g.updated_at, reverse=True)
        return {
            "hits": matches[:limit],
            "count": len(matches),
            "available": True,
            "error": None,
        }
    except Exception as exc:
        log.warning("search.goals.failed", error=str(exc))
        return {"hits": [], "count": 0, "available": True, "error": str(exc)}


def _search_trajectories(
    state: Any, q: str, q_uuid: UUID | None, limit: int
) -> dict[str, Any]:
    """Trajectories: meta-filter (id / goal_id / status substring) + step content scan.

    Используем `recent_trajectories(1000)` — для dev-объёмов хватает. Для
    содержательного поиска по шагам — `search_steps_by_terms` (keyword
    scoring). Возвращаем два списка: trajectories meta и step-hits.
    """
    ep = getattr(state, "episodic", None)
    if ep is None:
        return {"hits": [], "count": 0, "available": False, "error": None,
                "step_hits": [], "step_count": 0}
    try:
        q_lower = q.lower()
        # 1) Meta-level: rows = list of dicts with str id/goal_id/status.
        try:
            all_trajs = ep.recent_trajectories(limit=1000)
        except Exception as exc:
            log.warning("search.trajectories.recent_failed", error=str(exc))
            all_trajs = []
        matches: list[dict[str, Any]] = []
        for t in all_trajs:
            tid = str(t.get("id", ""))
            gid = str(t.get("goal_id", ""))
            status = str(t.get("status", ""))
            if (
                q_lower in tid.lower()
                or q_lower in gid.lower()
                or q_lower == status.lower()
            ):
                matches.append(t)
        # 2) Если q — UUID, попробуем exact-match как trajectory_id (это
        #    overlap с substring-логикой выше, но дешёво и явно).
        if q_uuid is not None and not any(
            str(t.get("id", "")) == str(q_uuid) for t in matches
        ):
            try:
                meta = ep.get_trajectory_meta(q_uuid)
                if meta is not None:
                    matches.insert(0, meta)
            except Exception:
                pass

        # 3) Step-level content search (только если q непустой).
        step_hits: list[dict[str, Any]] = []
        try:
            from harnes.memory.episodic import extract_terms

            terms = extract_terms(q)
            if terms:
                raw = ep.search_steps_by_terms(terms=terms, limit=limit)
                for r in raw:
                    try:
                        parsed = jsonlib.loads(r.get("content_json", "{}"))
                    except jsonlib.JSONDecodeError:
                        parsed = {"_raw": r.get("content_json", "")}
                    step_hits.append({**r, "content": parsed})
        except Exception as exc:
            log.warning("search.trajectories.steps_failed", error=str(exc))

        return {
            "hits": matches[:limit],
            "count": len(matches),
            "available": True,
            "error": None,
            "step_hits": step_hits,
            "step_count": len(step_hits),
        }
    except Exception as exc:
        log.warning("search.trajectories.failed", error=str(exc))
        return {"hits": [], "count": 0, "available": True, "error": str(exc),
                "step_hits": [], "step_count": 0}


def _search_journal(state: Any, q: str, limit: int) -> dict[str, Any]:
    """Journal: substring в payload_json (case-insensitive) + exact event_type.

    `recent_events(1000)` — последние 1000 события, потом filter. Для жирного
    журнала недостаточно; долгосрочный план — SQL LIKE прямо в SQLModel.
    """
    journal = getattr(state, "journal", None)
    if journal is None:
        return {"hits": [], "count": 0, "available": False, "error": None}
    try:
        from harnes.metacycle.journal import TickEventType

        q_lower = q.lower()
        # Exact match по event_type → отдельный запрос (точечный, без scan).
        et_match: list = []
        for et in TickEventType:
            if et.value.lower() == q_lower:
                try:
                    et_match = journal.recent_events(limit=limit, event_type=et)
                except Exception:
                    et_match = []
                break

        # Substring scan по payload — re-fetch 1000.
        try:
            recent = journal.recent_events(limit=1000)
        except Exception as exc:
            log.warning("search.journal.recent_failed", error=str(exc))
            recent = []
        substring_match = [
            e for e in recent if q_lower in (e.payload_json or "").lower()
        ]

        # Merge: et_match имеет приоритет (точное соответствие event_type).
        # Avoid дубликатов по id.
        seen_ids: set[int] = set()
        merged: list = []
        for e in et_match:
            if e.id is not None and e.id not in seen_ids:
                merged.append(e)
                seen_ids.add(e.id)
        for e in substring_match:
            if e.id is not None and e.id not in seen_ids:
                merged.append(e)
                seen_ids.add(e.id)

        return {
            "hits": merged[:limit],
            "count": len(merged),
            "available": True,
            "error": None,
        }
    except Exception as exc:
        log.warning("search.journal.failed", error=str(exc))
        return {"hits": [], "count": 0, "available": True, "error": str(exc)}


def _search_skills(state: Any, q: str, limit: int) -> dict[str, Any]:
    """Skills: substring в id / name / description (case-insensitive).

    `load_all()` читает все YAML-бандлы — это дёшево (десятки файлов).
    """
    registry = getattr(state, "skill_registry", None)
    if registry is None:
        return {"hits": [], "count": 0, "available": False, "error": None}
    try:
        q_lower = q.lower()
        matches = []
        for sk in registry.load_all():
            if (
                q_lower in sk.id.lower()
                or q_lower in sk.name.lower()
                or q_lower in (sk.description or "").lower()
            ):
                matches.append(sk)
        return {
            "hits": matches[:limit],
            "count": len(matches),
            "available": True,
            "error": None,
        }
    except Exception as exc:
        log.warning("search.skills.failed", error=str(exc))
        return {"hits": [], "count": 0, "available": True, "error": str(exc)}


def _search_eval(state: Any, q: str, limit: int) -> dict[str, Any]:
    """Eval runs: substring в adapter_name / eval_set / notes.

    include_held_out=True — search не должен скрывать «hidden» прогоны (это
    не leaderboard, это поиск). На UI можно отметить флажком.
    """
    eh = getattr(state, "eval_history", None)
    if eh is None:
        return {"hits": [], "count": 0, "available": False, "error": None}
    try:
        q_lower = q.lower()
        all_runs = eh.list_runs(include_held_out=True, limit=1000)
        matches = [
            r for r in all_runs
            if q_lower in (r.adapter_name or "").lower()
            or q_lower in (r.eval_set or "").lower()
            or q_lower in (r.notes or "").lower()
        ]
        return {
            "hits": matches[:limit],
            "count": len(matches),
            "available": True,
            "error": None,
        }
    except Exception as exc:
        log.warning("search.eval.failed", error=str(exc))
        return {"hits": [], "count": 0, "available": True, "error": str(exc)}


# ---------- Route ----------


@router.get("", response_class=HTMLResponse)
def universal_search(
    request: Request,
    q: str = "",
    types: str = "all",
    limit: int = 20,
) -> HTMLResponse:
    """Single-box search по всем stores.

    Параметры:
    - q: текст запроса. Пусто → пустые результаты (не делаем "list all rows").
    - types: CSV ("goals,trajectories,journal,skills,eval") или "all".
    - limit: per-source cap, max 100.

    Возвращаемая структура per source:
        {"hits": [...], "count": N, "available": bool, "error": str | None}
    + total + breakdown.

    UUID-shortcut: если q валидный UUID — пробуем exact `goal_repo.get(UUID)` и
    `episodic.get_trajectory_meta(UUID)`. Это дешевле, чем scan'ить все цели.
    """
    limit = max(1, min(limit, 100))
    selected_types = _parse_types(types)
    q_clean = (q or "").strip()
    q_uuid = _try_uuid(q_clean) if q_clean else None

    state = request.app.state
    results: dict[str, dict[str, Any]] = {}
    total = 0

    # Пустой query — возвращаем пустые результаты (per spec).
    if not q_clean:
        for src in ALL_SOURCES:
            # Прокидываем availability, чтобы UI мог honestly сказать
            # "skills off" даже до первого поиска.
            available = getattr(state, _state_attr(src), None) is not None
            results[src] = {
                "hits": [], "count": 0, "available": available, "error": None,
                **({"step_hits": [], "step_count": 0} if src == "trajectories" else {}),
            }
    else:
        if "goals" in selected_types:
            results["goals"] = _search_goals(state, q_clean, q_uuid, limit)
        if "trajectories" in selected_types:
            results["trajectories"] = _search_trajectories(state, q_clean, q_uuid, limit)
        if "journal" in selected_types:
            results["journal"] = _search_journal(state, q_clean, limit)
        if "skills" in selected_types:
            results["skills"] = _search_skills(state, q_clean, limit)
        if "eval" in selected_types:
            results["eval"] = _search_eval(state, q_clean, limit)

        # Source'ы, которые operator снял чекбоксом — кладём placeholder, чтобы
        # шаблон мог отрисовать "0 hits / off" вместо KeyError.
        for src in ALL_SOURCES:
            if src not in results:
                available = getattr(state, _state_attr(src), None) is not None
                results[src] = {
                    "hits": [], "count": 0, "available": available, "error": None,
                    **({"step_hits": [], "step_count": 0} if src == "trajectories" else {}),
                }

        total = sum(r.get("count", 0) for r in results.values())
        # Trajectory steps count тоже в total — это отдельный класс хитов.
        total += results.get("trajectories", {}).get("step_count", 0)

    return templates.TemplateResponse(
        request,
        "search/results.html",
        {
            "q": q,
            "q_clean": q_clean,
            "q_is_uuid": q_uuid is not None,
            "types_csv": types,
            "selected_types": selected_types,
            "all_sources": list(ALL_SOURCES),
            "limit": limit,
            "results": results,
            "total": total,
        },
    )


def _state_attr(source: str) -> str:
    """Mapping source-name → app.state атрибут (для availability-check)."""
    return {
        "goals": "goal_repo",
        "trajectories": "episodic",
        "journal": "journal",
        "skills": "skill_registry",
        "eval": "eval_history",
    }[source]
