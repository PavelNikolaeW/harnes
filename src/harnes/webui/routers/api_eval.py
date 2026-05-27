"""Eval-history view — список прогонов benchmark'а + side-by-side сравнение.

См. CLI команды eval-history / eval-compare (operator/cli.py). По дефолту
held-out прогоны скрыты — research-hygiene из v1.0 #31.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.eval import EvalHistoryStore, EvalRunRecord
from harnes.webui.deps import get_eval_history
from harnes.webui.templating import templates

router = APIRouter()


# ---------- helpers ----------


def _diff_rows(base: EvalRunRecord, cand: EvalRunRecord) -> list[dict[str, Any]]:
    """Список метрик с base/cand/delta/format — для рендера в таблице сравнения."""

    def _row(label: str, b: float, c: float, fmt: str = "pct") -> dict[str, Any]:
        delta = c - b
        return {
            "label": label,
            "base": b,
            "cand": c,
            "delta": delta,
            "abs_delta": abs(delta),
            "fmt": fmt,
            "arrow": "↑" if delta > 0 else ("↓" if delta < 0 else "="),
            "direction": (
                "up" if delta > 0 else ("down" if delta < 0 else "flat")
            ),
        }

    return [
        _row("success_rate",    base.success_rate,    cand.success_rate),
        _row(f"pass@{base.repeat_k}",  base.pass_at_k,    cand.pass_at_k),
        _row(f"stable@{base.repeat_k}", base.stable_at_k, cand.stable_at_k),
        _row("avg_steps",       base.avg_steps,       cand.avg_steps,       "num2"),
        _row("p50_steps",       base.p50_steps,       cand.p50_steps,       "num2"),
        _row("p95_steps",       base.p95_steps,       cand.p95_steps,       "num2"),
        _row("p50_latency_s",   base.p50_latency_s,   cand.p50_latency_s,   "sec"),
        _row("p95_latency_s",   base.p95_latency_s,   cand.p95_latency_s,   "sec"),
        _row("failure_entropy", base.failure_entropy, cand.failure_entropy, "num2"),
        _row("avg_tokens",      base.avg_cost_tokens, cand.avg_cost_tokens, "num0"),
    ]


def _comparison_warnings(base: EvalRunRecord, cand: EvalRunRecord) -> list[str]:
    """Когда сравнение по факту бессмысленно — флагать оператору."""
    out: list[str] = []
    if base.adapter_name != cand.adapter_name:
        out.append(f"adapter_name: {base.adapter_name} vs {cand.adapter_name}")
    if (base.eval_set or "") != (cand.eval_set or ""):
        out.append(
            f"eval_set: {base.eval_set or '(none)'} vs {cand.eval_set or '(none)'}"
        )
    elif base.eval_set_hash and base.eval_set_hash != cand.eval_set_hash:
        out.append(
            f"eval_set_hash: {base.eval_set_hash} vs {cand.eval_set_hash} "
            "(тот же ярлык, но РАЗНЫЕ task'и)"
        )
    if base.repeat_k != cand.repeat_k:
        out.append(f"repeat_k: {base.repeat_k} vs {cand.repeat_k}")
    if base.held_out != cand.held_out:
        out.append(f"held_out: {base.held_out} vs {cand.held_out}")
    return out


# ---------- pages ----------


@router.get("", response_class=HTMLResponse)
def list_eval_runs(
    request: Request,
    adapter: str | None = None,
    eval_set: str | None = None,
    include_held_out: bool = False,
    limit: int = 50,
    store: EvalHistoryStore = Depends(get_eval_history),
) -> HTMLResponse:
    """Recent N прогонов с фильтрами."""
    limit = max(1, min(limit, 500))
    runs = store.list_runs(
        adapter_name=adapter,
        eval_set=eval_set,
        include_held_out=include_held_out,
        limit=limit,
    )
    return templates.TemplateResponse(
        request,
        "eval/list.html",
        {
            "runs": runs,
            "adapter": adapter,
            "eval_set": eval_set,
            "include_held_out": include_held_out,
            "limit": limit,
        },
    )


@router.get("/compare", response_class=HTMLResponse)
def compare_runs(
    request: Request,
    base: int,
    cand: int | None = None,
    store: EvalHistoryStore = Depends(get_eval_history),
) -> HTMLResponse:
    """Side-by-side метрик baseline → candidate.

    Если cand не указан — берётся latest того же adapter'а (как в CLI).
    """
    b = store.get(base)
    if b is None:
        raise HTTPException(404, f"baseline run #{base} not found")
    if cand is None:
        latest = store.latest(adapter_name=b.adapter_name)
        if latest is None or latest.id == b.id:
            raise HTTPException(400, "no newer candidate run for this adapter")
        c = latest
    else:
        c = store.get(cand)
        if c is None:
            raise HTTPException(404, f"candidate run #{cand} not found")

    base_modes = json.loads(b.failure_modes_json or "{}")
    cand_modes = json.loads(c.failure_modes_json or "{}")
    mode_rows = []
    for m in sorted(set(base_modes) | set(cand_modes)):
        b_v = int(base_modes.get(m, 0))
        c_v = int(cand_modes.get(m, 0))
        mode_rows.append({
            "mode": m,
            "base": b_v,
            "cand": c_v,
            "delta": c_v - b_v,
            "arrow": "↑" if c_v > b_v else ("↓" if c_v < b_v else "="),
        })

    base_skills = json.loads(b.skill_versions_json or "{}")
    cand_skills = json.loads(c.skill_versions_json or "{}")
    skill_diffs = []
    for k in sorted(set(base_skills) | set(cand_skills)):
        bv, cv = base_skills.get(k, "—"), cand_skills.get(k, "—")
        if bv != cv:
            skill_diffs.append({"skill_id": k, "base": bv, "cand": cv})

    return templates.TemplateResponse(
        request,
        "eval/compare.html",
        {
            "base": b,
            "cand": c,
            "rows": _diff_rows(b, c),
            "warnings": _comparison_warnings(b, c),
            "mode_rows": mode_rows,
            "skill_diffs": skill_diffs,
        },
    )


@router.get("/{run_id}", response_class=HTMLResponse)
def eval_detail(
    request: Request,
    run_id: int,
    store: EvalHistoryStore = Depends(get_eval_history),
) -> HTMLResponse:
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, f"run #{run_id} not found")

    failure_modes = json.loads(run.failure_modes_json or "{}")
    skill_versions = json.loads(run.skill_versions_json or "{}")

    return templates.TemplateResponse(
        request,
        "eval/detail.html",
        {
            "run": run,
            "failure_modes": failure_modes,
            "skill_versions": skill_versions,
        },
    )
