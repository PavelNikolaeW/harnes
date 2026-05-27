"""Self-critic для ReAct loop'а.

См. `agent_architecture.html` § 11.

Срабатывает перед action'ом с `is_irreversible=true`. Отдельный LLM-вызов
с ролью критика (tier=light, анти-коррелирован с executor). Возвращает
CritiqueStep с verdict ok / warning / reject:

- ok      → action исполняется
- warning → action исполняется, verdict копится метрикой
- reject  → action НЕ исполняется, цикл возвращается к thought

Лимит rejections per trajectory предотвращает зацикливание.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable
from uuid import UUID

import structlog

from harnes.goals.schema import Goal
from harnes.react.schema import (
    ActionStep,
    Cost,
    CritiqueStep,
    CritiqueVerdict,
    Trajectory,
)

log = structlog.get_logger()


_CRITIC_SYSTEM_PROMPT = (
    "You are a careful critic evaluating a single proposed irreversible action "
    "before it is executed. You are not the agent — you are an independent check. "
    "Focus narrowly on: (a) does this action match what the goal requires, "
    "(b) will it destroy/overwrite/lose anything valuable, (c) is there a "
    "cheaper/safer alternative that should be tried first. "
    "Reply with strict JSON only, no other text."
)


def _format_recent_steps(trajectory: Trajectory, n: int = 4) -> str:
    """Краткая выжимка последних n шагов траектории — контекст для критика."""
    recent = trajectory.steps[-n:]
    lines: list[str] = []
    for s in recent:
        if s.type == "thought":
            lines.append(f"  thought: {s.text[:200]}")
        elif s.type == "action":
            lines.append(f"  action: {s.tool_id}({json.dumps(s.args)[:200]})")
        elif s.type == "observation":
            payload = (
                json.dumps(s.payload)[:200] if s.payload else (s.error_detail or "")
            )
            lines.append(f"  observation[{s.outcome.value}]: {payload}")
        else:
            lines.append(f"  {s.type}")
    return "\n".join(lines) if lines else "  (no prior steps)"


def _parse_critique_json(text: str) -> dict[str, Any] | None:
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


def critique_action(
    action: ActionStep,
    trajectory: Trajectory,
    goal: Goal,
    llm_call: Callable[..., Any] | None = None,
) -> CritiqueStep:
    """Критикует предложенный action. Возвращает CritiqueStep с verdict.

    При любых ошибках LLM-вызова или парсинга — verdict=ok с пометкой в
    reasoning. **Не блокируем исполнение из-за сбоя критика** — это
    защитная мера, не блокирующее звено.
    """
    if llm_call is None:
        from harnes.llm import call as default_call

        llm_call = default_call

    user_prompt = f"""Goal: {goal.description}

Proposed irreversible action:
  tool_id: {action.tool_id}
  args:    {json.dumps(action.args)}

Recent trajectory context:
{_format_recent_steps(trajectory, n=4)}

Evaluate strictly. Reply with JSON only:
{{
  "verdict": "ok" | "warning" | "reject",
  "reasoning": "<one short sentence>",
  "risks": ["<short>", ...],
  "recommendation": "<only if reject — what to do instead, else empty string>"
}}"""

    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # max_tokens=1500 для thinking + JSON verdict. См. llm/client.py.
        response = llm_call(messages, tier="light", max_tokens=1500)
        text = response.choices[0].message.content or ""
        tokens = getattr(response.usage, "completion_tokens", 0) or 0
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "react.critic.llm_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        # Не блокируем исполнение — fall-open с ok-verdict.
        return CritiqueStep(
            target_step_id=action.id,
            verdict=CritiqueVerdict.OK,
            reasoning=f"critic LLM failed ({type(exc).__name__}); falling open",
            cost=Cost(tokens=0),
        )

    parsed = _parse_critique_json(text)
    if parsed is None:
        log.warning("react.critic.unparseable", raw=text[:200])
        return CritiqueStep(
            target_step_id=action.id,
            verdict=CritiqueVerdict.OK,
            reasoning="critic returned unparseable JSON; falling open",
            cost=Cost(tokens=tokens),
        )

    verdict_raw = str(parsed.get("verdict", "ok")).lower()
    try:
        verdict = CritiqueVerdict(verdict_raw)
    except ValueError:
        log.warning("react.critic.unknown_verdict", verdict=verdict_raw)
        verdict = CritiqueVerdict.OK

    reasoning = str(parsed.get("reasoning", "")).strip()
    risks = parsed.get("risks", []) or []
    recommendation = str(parsed.get("recommendation", "")).strip() or None

    log.info(
        "react.critic.done",
        target_step_id=str(action.id),
        verdict=verdict.value,
        tool_id=action.tool_id,
    )

    return CritiqueStep(
        target_step_id=action.id,
        verdict=verdict,
        reasoning=reasoning,
        recommendation=recommendation,
        risks_identified=[str(r) for r in risks],
        cost=Cost(tokens=tokens),
    )
