"""Verifiers — реализация verify-стадии метацикла per predicate type.

См. `agent_architecture.html` § 3 (verify), § 4 (predicate types).

Этап verify в tick.py делегирует сюда по типу предиката цели:

- structural    → verify_structural
- judge         → verify_judge  (LLM-судья, tier=light, анти-коррелирован)
- state_change  → verify_state_change  (TBD — нужен check_tool из ToolRegistry)
- composite     → verify_composite  (TBD — агрегация над детьми)
- external      → verify_external  (всегда UNDETERMINED + регистрация в pending)
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import structlog

from harnes.goals.schema import (
    CompositePredicate,
    ExternalPredicate,
    Goal,
    JudgePredicate,
    StateChangePredicate,
    StructuralPredicate,
)
from harnes.metacycle.schema import Verdict, VerifyStatus
from harnes.react.schema import Trajectory

log = structlog.get_logger()


# ---------- structural ----------


def verify_structural(
    predicate: StructuralPredicate,
    trajectory: Trajectory,
    goal: Goal,
) -> Verdict:
    """v0: проверка минимальная — final_state должен быть.

    TODO: настоящая JSON-Schema валидация final_state vs predicate.expected_schema.
    """
    if trajectory.final_state is None:
        return Verdict(
            status=VerifyStatus.FAIL,
            reasons=["final_state missing"],
            measured_by="structural",
        )
    return Verdict(
        status=VerifyStatus.SUCCESS,
        reasons=["final_state present"],
        measured_by="structural",
    )


# ---------- judge ----------


_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating whether an autonomous agent succeeded "
    "at a stated goal. You have no role in the agent's reasoning — evaluate "
    "strictly against the stated criterion, using only the evidence provided. "
    "If evidence is insufficient, return success=false with reasoning explaining "
    "what's missing. Reply with strict JSON only, no other text."
)


def _format_recent_steps(trajectory: Trajectory, n: int = 5) -> str:
    """Краткое описание последних n шагов траектории для judge'а."""
    recent = trajectory.steps[-n:]
    lines = []
    for s in recent:
        if s.type == "thought":
            lines.append(f"  thought: {s.text[:200]}")
        elif s.type == "action":
            lines.append(f"  action: {s.tool_id}({json.dumps(s.args)[:200]})")
        elif s.type == "observation":
            payload = json.dumps(s.payload)[:200] if s.payload else (s.error_detail or "")
            lines.append(f"  observation[{s.outcome.value}]: {payload}")
        else:
            lines.append(f"  {s.type}")
    return "\n".join(lines) if lines else "  (no steps)"


def _parse_judge_json(text: str) -> dict[str, Any] | None:
    """Достаёт первый JSON-объект из ответа judge'а."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def verify_judge(
    predicate: JudgePredicate,
    trajectory: Trajectory,
    goal: Goal,
    llm_call: Callable[..., Any] | None = None,
) -> Verdict:
    """LLM-судья оценивает соответствие траектории criterion'у.

    Использует tier=light с явной анти-коррелированной ролью «судья».
    Возвращает UNDETERMINED при любых ошибках вызова или парсинга —
    не FAIL, потому что мы не можем уверенно сказать «не достигнуто».
    """
    if llm_call is None:
        from harnes.llm import call as default_call

        llm_call = default_call

    final_state_text = (
        json.dumps(trajectory.final_state) if trajectory.final_state is not None else "(none)"
    )

    user_prompt = f"""Goal: {goal.description}

Success criterion: {predicate.criterion}

Agent's final state: {final_state_text}

Last steps of the agent's trajectory:
{_format_recent_steps(trajectory, n=5)}

Strictly evaluate: did the agent satisfy the success criterion?

Reply with JSON only:
{{"success": true|false, "reasoning": "<one or two sentences>"}}"""

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = llm_call(messages, tier="light", max_tokens=300)
        text = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        log.error(
            "verify.judge.llm_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return Verdict(
            status=VerifyStatus.UNDETERMINED,
            reasons=[f"judge LLM invocation failed: {type(exc).__name__}: {exc}"],
            measured_by="judge_llm",
        )

    parsed = _parse_judge_json(text)
    if parsed is None:
        log.warning("verify.judge.unparseable", raw=text[:200])
        return Verdict(
            status=VerifyStatus.UNDETERMINED,
            reasons=["judge LLM returned unparseable output"],
            evidence=[{"raw_output": text[:500]}],
            measured_by="judge_llm",
        )

    success = bool(parsed.get("success", False))
    reasoning = str(parsed.get("reasoning", "")).strip()

    log.info(
        "verify.judge.done",
        goal_id=str(goal.id),
        success=success,
        reasoning=reasoning[:200],
    )

    return Verdict(
        status=VerifyStatus.SUCCESS if success else VerifyStatus.FAIL,
        reasons=[reasoning] if reasoning else [],
        evidence=[{"judge_raw": text[:500]}],
        measured_by="judge_llm",
    )


# ---------- state_change ----------


def verify_state_change(
    predicate: StateChangePredicate,
    trajectory: Trajectory,
    goal: Goal,
) -> Verdict:
    """v0: stub — нужен ToolRegistry для вызова check_tool. Подключим в v0.2."""
    return Verdict(
        status=VerifyStatus.UNDETERMINED,
        reasons=["state_change verifier not yet implemented (v0.2)"],
        measured_by="state_change",
    )


# ---------- composite ----------


def verify_composite(
    predicate: CompositePredicate,
    trajectory: Trajectory,
    goal: Goal,
) -> Verdict:
    """v0: stub — нужен goal_repo для обхода children. Подключим в v0.2."""
    return Verdict(
        status=VerifyStatus.UNDETERMINED,
        reasons=["composite verifier not yet implemented (v0.2)"],
        measured_by="composite",
    )


# ---------- external ----------


def verify_external(
    predicate: ExternalPredicate,
    trajectory: Trajectory,
    goal: Goal,
) -> Verdict:
    """External-предикаты всегда deferred. Регистрация в pending_verifications —
    отдельная ответственность tick.py (через goal_repo)."""
    return Verdict(
        status=VerifyStatus.UNDETERMINED,
        reasons=[
            f"external predicate — awaits signal {predicate.expected_signal!r}"
        ],
        measured_by="external",
    )


# ---------- dispatcher ----------


def verify(
    trajectory: Trajectory,
    goal: Goal,
    llm_call: Callable[..., Any] | None = None,
) -> Verdict:
    """Главный entry-point — диспетчер по типу предиката."""
    predicate = goal.predicate_of_success
    if isinstance(predicate, StructuralPredicate):
        return verify_structural(predicate, trajectory, goal)
    if isinstance(predicate, JudgePredicate):
        return verify_judge(predicate, trajectory, goal, llm_call=llm_call)
    if isinstance(predicate, StateChangePredicate):
        return verify_state_change(predicate, trajectory, goal)
    if isinstance(predicate, CompositePredicate):
        return verify_composite(predicate, trajectory, goal)
    if isinstance(predicate, ExternalPredicate):
        return verify_external(predicate, trajectory, goal)
    # Should be unreachable — discriminated union covers all cases.
    return Verdict(
        status=VerifyStatus.UNDETERMINED,
        reasons=[f"unknown predicate type: {type(predicate).__name__}"],
        measured_by="unknown",
    )
