"""Diff Trajectories — side-by-side сравнение двух trajectory'ев.

Use case: оператор побежал ту же цель до и после bump'а скилла через reflect,
хочет увидеть что поменялось — какие шаги изменились, метрики, final state.

Mounted под `/trajectories-diff`. Endpoint: `GET /diff?left={uuid}&right={uuid}`.
"""
from __future__ import annotations

import difflib
import json
from itertools import zip_longest
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.goals.store import GoalRepository
from harnes.memory.episodic import EpisodicStore
from harnes.webui.deps import get_episodic, get_goal_repo
from harnes.webui.templating import templates

router = APIRouter()


# ---------- helpers ----------


def _parse_content(content_json: str) -> dict[str, Any]:
    """LanceDB хранит content как JSON-строку. None-safe."""
    if not content_json:
        return {}
    try:
        return json.loads(content_json)
    except json.JSONDecodeError:
        return {"_raw": content_json}


def _normalize_step(raw: dict[str, Any]) -> dict[str, Any]:
    """Прикручиваем parsed `content` и оставляем сырые поля."""
    return {
        **raw,
        "content": _parse_content(raw.get("content_json", "")),
    }


def _aggregate(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Подсчёт total_tokens, total_latency, breakdown по типам."""
    type_counts: dict[str, int] = {}
    total_tokens = 0
    total_latency = 0.0
    for s in steps:
        t = s.get("step_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        total_tokens += int(s.get("cost_tokens") or 0)
        total_latency += float(s.get("cost_latency") or 0.0)
    return {
        "step_count": len(steps),
        "total_tokens": total_tokens,
        "total_latency": total_latency,
        "type_counts": type_counts,
    }


def _canonical_json(value: Any) -> str:
    """Стабильный JSON-канон для сравнения content dict'ов (отсортированные ключи)."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _align_steps(
    left_steps: list[dict[str, Any]],
    right_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align по index (zip_longest). При расхождении длин — None-placeholder.

    На каждой паре отмечаем `same_type` (типы шагов совпали) и `content_match`
    (сериализованный content идентичен).
    """
    pairs: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(zip_longest(left_steps, right_steps), start=1):
        same_type = (
            left is not None
            and right is not None
            and left.get("step_type") == right.get("step_type")
        )
        content_match = (
            left is not None
            and right is not None
            and _canonical_json(left.get("content"))
            == _canonical_json(right.get("content"))
        )
        pairs.append(
            {
                "index": idx,
                "left_step": left,
                "right_step": right,
                "same_type": same_type,
                "content_match": content_match,
            }
        )
    return pairs


def _metric_rows(
    left_agg: dict[str, Any],
    right_agg: dict[str, Any],
    left_meta: dict[str, Any],
    right_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Метрик side-by-side с delta/arrow/direction (как в eval/compare)."""

    def _row(label: str, l_v: float, r_v: float, fmt: str = "num0") -> dict[str, Any]:
        delta = r_v - l_v
        return {
            "label": label,
            "left": l_v,
            "right": r_v,
            "delta": delta,
            "abs_delta": abs(delta),
            "fmt": fmt,
            "arrow": "↑" if delta > 0 else ("↓" if delta < 0 else "="),
            "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
        }

    return [
        _row("steps", left_agg["step_count"], right_agg["step_count"], "num0"),
        _row("total_tokens", left_agg["total_tokens"], right_agg["total_tokens"], "num0"),
        _row("total_latency", left_agg["total_latency"], right_agg["total_latency"], "sec"),
    ]


def _final_state_value(meta: dict[str, Any]) -> tuple[Any, str]:
    """Разбираем final_state_json. Возвращает (value, kind), kind ∈ {'json','text','empty'}."""
    raw = meta.get("final_state_json") or ""
    if not raw:
        return None, "empty"
    try:
        return json.loads(raw), "json"
    except (json.JSONDecodeError, TypeError):
        return raw, "text"


def _diff_final_state(
    left_value: Any,
    left_kind: str,
    right_value: Any,
    right_kind: str,
) -> dict[str, Any]:
    """Если оба JSON-dict — сравниваем ключи. Иначе — unified text diff."""
    result: dict[str, Any] = {
        "left_value": left_value,
        "right_value": right_value,
        "left_kind": left_kind,
        "right_kind": right_kind,
        "added": [],
        "removed": [],
        "changed": [],
        "text_diff": None,
        "kind_mismatch": False,
    }

    # Оба пустые — diff не нужен.
    if left_kind == "empty" and right_kind == "empty":
        return result

    # Разные kinds (json vs text vs empty) — флагаем + дамперим text-diff.
    if left_kind != right_kind:
        result["kind_mismatch"] = True
        left_text = (
            json.dumps(left_value, indent=2, ensure_ascii=False, default=str)
            if left_kind == "json"
            else (left_value or "")
        )
        right_text = (
            json.dumps(right_value, indent=2, ensure_ascii=False, default=str)
            if right_kind == "json"
            else (right_value or "")
        )
        result["text_diff"] = "\n".join(
            difflib.unified_diff(
                left_text.splitlines(),
                right_text.splitlines(),
                fromfile="left",
                tofile="right",
                lineterm="",
            )
        )
        return result

    # Оба JSON. Сравниваем ключи на верхнем уровне (если оба dict).
    if (
        left_kind == "json"
        and isinstance(left_value, dict)
        and isinstance(right_value, dict)
    ):
        left_keys = set(left_value.keys())
        right_keys = set(right_value.keys())
        result["added"] = sorted(right_keys - left_keys)
        result["removed"] = sorted(left_keys - right_keys)
        for k in sorted(left_keys & right_keys):
            if _canonical_json(left_value[k]) != _canonical_json(right_value[k]):
                result["changed"].append(
                    {"key": k, "left": left_value[k], "right": right_value[k]}
                )
        return result

    # Оба JSON, но хотя бы один — не dict (list/str/num). Просто line-diff.
    if left_kind == "json":
        left_text = json.dumps(left_value, indent=2, ensure_ascii=False, default=str)
        right_text = json.dumps(right_value, indent=2, ensure_ascii=False, default=str)
    else:  # text
        left_text = left_value or ""
        right_text = right_value or ""

    result["text_diff"] = "\n".join(
        difflib.unified_diff(
            left_text.splitlines(),
            right_text.splitlines(),
            fromfile="left",
            tofile="right",
            lineterm="",
        )
    )
    return result


def _safe_goal(goal_repo: GoalRepository, goal_id_str: str | None) -> Any:
    if not goal_id_str:
        return None
    try:
        return goal_repo.get(UUID(goal_id_str))
    except (ValueError, Exception):
        return None


# ---------- endpoint ----------


@router.get("/diff", response_class=HTMLResponse)
def diff_trajectories(
    request: Request,
    left: str,
    right: str,
    episodic: EpisodicStore = Depends(get_episodic),
    goal_repo: GoalRepository = Depends(get_goal_repo),
) -> HTMLResponse:
    """Side-by-side render двух trajectory'ев."""
    # 1. Validate UUIDs.
    try:
        left_id = UUID(left)
    except (ValueError, TypeError):
        raise HTTPException(400, f"invalid left trajectory id: {left}")
    try:
        right_id = UUID(right)
    except (ValueError, TypeError):
        raise HTTPException(400, f"invalid right trajectory id: {right}")

    # 2. Fetch meta + steps.
    left_meta = episodic.get_trajectory_meta(left_id)
    if left_meta is None:
        raise HTTPException(404, f"trajectory {left} not found")
    right_meta = episodic.get_trajectory_meta(right_id)
    if right_meta is None:
        raise HTTPException(404, f"trajectory {right} not found")

    left_steps_raw = episodic.get_steps(left_id)
    right_steps_raw = episodic.get_steps(right_id)
    left_steps = [_normalize_step(s) for s in left_steps_raw]
    right_steps = [_normalize_step(s) for s in right_steps_raw]

    # 3. Goals.
    left_goal = _safe_goal(goal_repo, left_meta.get("goal_id"))
    right_goal = _safe_goal(goal_repo, right_meta.get("goal_id"))
    same_goal = (
        bool(left_meta.get("goal_id"))
        and left_meta.get("goal_id") == right_meta.get("goal_id")
    )

    # 4. Aggregates.
    left_agg = _aggregate(left_steps_raw)
    right_agg = _aggregate(right_steps_raw)

    # 5. Align.
    pairs = _align_steps(left_steps, right_steps)

    # 6. Final state diff.
    left_fs, left_kind = _final_state_value(left_meta)
    right_fs, right_kind = _final_state_value(right_meta)
    final_diff = _diff_final_state(left_fs, left_kind, right_fs, right_kind)

    # 7. Metric rows.
    metric_rows = _metric_rows(left_agg, right_agg, left_meta, right_meta)

    # 8. Warnings (goal mismatch / status mismatch и т.п. оператор сам решит).
    warnings: list[str] = []
    if not same_goal:
        warnings.append(
            f"different goals: left={left_meta.get('goal_id', '—')}, "
            f"right={right_meta.get('goal_id', '—')}"
        )
    if final_diff["kind_mismatch"]:
        warnings.append(
            f"final_state kind mismatch: left={left_kind}, right={right_kind}"
        )

    return templates.TemplateResponse(
        request,
        "diff/trajectories.html",
        {
            "left_meta": left_meta,
            "right_meta": right_meta,
            "left_steps": left_steps,
            "right_steps": right_steps,
            "left_agg": left_agg,
            "right_agg": right_agg,
            "left_goal": left_goal,
            "right_goal": right_goal,
            "same_goal": same_goal,
            "pairs": pairs,
            "metric_rows": metric_rows,
            "final_diff": final_diff,
            "warnings": warnings,
            "left_id": str(left_id),
            "right_id": str(right_id),
        },
    )
