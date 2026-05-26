"""Reflect — стадия самоулучшения метацикла.

См. `agent_architecture.html` § 15.

В v0.2 реализован один режим:

- **failure_analysis** — при verify=FAIL для траектории LLM-судья (tier=main)
  анализирует, что пошло не так, и предлагает обновление prompt_template
  использованного скилла. Если предложение конструктивное — SkillRegistry
  создаёт новую версию скилла; старая остаётся в metrics-таблице.

Прочие режимы (batch_consolidation, periodic_review, ...) — v0.3+.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Callable

import structlog

from harnes.goals.schema import Goal
from harnes.metacycle.schema import Verdict, VerifyStatus
from harnes.react.schema import Trajectory
from harnes.skills.schema import Skill, SkillStatus
from harnes.skills.store import SkillRegistry

log = structlog.get_logger()


_FAILURE_ANALYSIS_SYSTEM_PROMPT = (
    "You are an analyst diagnosing a failed agent trajectory. Your job is to "
    "identify what went wrong and propose a concrete improvement to the agent's "
    "prompt template that would have prevented this failure. Be specific. "
    "If no clear prompt-level fix exists (e.g., failure was environmental), "
    "respond with should_update=false. Reply with strict JSON only."
)


def _format_trajectory_for_reflect(traj: Trajectory, max_steps: int = 10) -> str:
    """Краткое описание последних шагов траектории."""
    recent = traj.steps[-max_steps:]
    lines: list[str] = [f"status: {traj.status.value if traj.status else 'unknown'}"]
    for i, s in enumerate(recent, 1):
        if s.type == "thought":
            lines.append(f"  [{i}] thought: {s.text[:200]}")
        elif s.type == "action":
            args_str = json.dumps(s.args)[:200]
            lines.append(f"  [{i}] action: {s.tool_id}({args_str})")
        elif s.type == "observation":
            payload = (
                json.dumps(s.payload)[:200] if s.payload else (s.error_detail or "")
            )
            lines.append(f"  [{i}] observation[{s.outcome.value}]: {payload}")
        elif s.type == "critique":
            lines.append(f"  [{i}] critique[{s.verdict.value}]: {s.reasoning[:200]}")
        else:
            lines.append(f"  [{i}] {s.type}")
    return "\n".join(lines)


def _parse_reflect_json(text: str) -> dict[str, Any] | None:
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


def _bump_patch_version(version: str) -> str:
    """0.0.1 → 0.0.2; 1.2.3 → 1.2.4. Если строка не семвер — добавляем '+r1'."""
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if m:
        major, minor, patch = (int(m.group(i)) for i in (1, 2, 3))
        return f"{major}.{minor}.{patch + 1}"
    return f"{version}+r1"


def reflect_failure_analysis(
    trajectory: Trajectory,
    goal: Goal,
    verdict: Verdict,
    skill: Skill,
    skill_registry: SkillRegistry,
    llm_call: Callable[..., Any] | None = None,
) -> Skill | None:
    """Анализирует failed-траекторию; предлагает новый prompt_template скилла.

    Возвращает Skill новой версии (уже сохранённый в registry) или None
    если изменений не нужно.

    Fall-open: при ошибках LLM или парсинга возвращает None, не блокирует
    метацикл.
    """
    if llm_call is None:
        from harnes.llm import call as default_call

        llm_call = default_call

    user_prompt = f"""Failed trajectory analysis.

Goal: {goal.description}
Goal class: {goal.goal_class.value}
Predicate criterion (or type): {goal.predicate_of_success.type}

Verdict: {verdict.status.value} (measured_by={verdict.measured_by})
Verdict reasons: {'; '.join(verdict.reasons) if verdict.reasons else '(none)'}

Skill used: {skill.id} (version {skill.version})
Current prompt template:
---
{skill.prompt_template}
---

Trajectory:
{_format_trajectory_for_reflect(trajectory)}

Analyze:
1. What specifically went wrong?
2. Would a change to the prompt template have prevented this failure?
3. If yes, write the FULL updated prompt template (keep placeholders like
   {{goal_description}} and {{tools_list}} unchanged).

Reply with JSON only:
{{
  "diagnosis": "<one short paragraph>",
  "should_update": true | false,
  "new_prompt_template": "<full updated template, or empty string if not updating>"
}}"""

    messages = [
        {"role": "system", "content": _FAILURE_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # main tier — у нас в dev gemma-26b-a4b, в проде gemma-31b-mtp
        response = llm_call(messages, tier="main", max_tokens=2000)
        text = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "reflect.failure_analysis.llm_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    parsed = _parse_reflect_json(text)
    if parsed is None:
        log.warning("reflect.failure_analysis.unparseable", raw=text[:300])
        return None

    diagnosis = str(parsed.get("diagnosis", "")).strip()
    should_update = bool(parsed.get("should_update", False))
    new_template = str(parsed.get("new_prompt_template", "")).strip()

    log.info(
        "reflect.failure_analysis.done",
        skill_id=skill.id,
        should_update=should_update,
        diagnosis=diagnosis[:200],
    )

    if not should_update or not new_template or new_template == skill.prompt_template:
        return None

    # Sanity: должны сохраниться ключевые плейсхолдеры.
    for placeholder in ("{goal_description}", "{tools_list}"):
        if placeholder in skill.prompt_template and placeholder not in new_template:
            log.warning(
                "reflect.failure_analysis.template_missing_placeholder",
                placeholder=placeholder,
            )
            return None

    new_version = _bump_patch_version(skill.version)
    new_skill = skill.model_copy(
        update={
            "version": new_version,
            "parent_version_id": skill.version,
            "prompt_template": new_template,
            "updated_at": datetime.now(UTC),
            "status": SkillStatus.ACTIVE,
        }
    )

    skill_registry.save(new_skill)
    log.info(
        "reflect.skill.versioned",
        skill_id=skill.id,
        from_version=skill.version,
        to_version=new_version,
    )
    return new_skill
