"""Reflect — стадия самоулучшения метацикла.

См. `agent_architecture.html` § 15.

Режимы:

- **failure_analysis** (v0.2) — при verify=FAIL LLM-судья (tier=main)
  анализирует, что пошло не так, и предлагает обновление prompt_template
  использованного скилла. Если предложение конструктивное — SkillRegistry
  создаёт новую версию скилла.

- **inquiry_from_failure** (v1.0 #33) — при verify=FAIL LLM решает, есть
  ли в траектории явный знание-gap, который стоит проактивно закрыть.
  Если есть — spawn'ит inquiry-goal с origin=SELF, origin_subtype=LEARNING.
  Это первый кирпич автономии — агент сам генерирует себе вопросы.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Callable

import structlog

from harnes.goals.schema import (
    Goal,
    GoalClass,
    JudgePredicate,
    Origin,
    OriginSubtype,
)
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


# ---------- Inquiry from failure (v1.0 #33) ----------


_INQUIRY_SYSTEM_PROMPT = (
    "You are an analyst looking for knowledge gaps in a failed agent trajectory. "
    "Decide whether the failure stems from missing information that could be "
    "investigated as an independent inquiry. Examples of inquiry-worthy gaps: "
    "an unfamiliar API, a poorly-understood error pattern, a missing fact about "
    "the environment. NOT inquiry-worthy: prompt-template mistakes (those go to "
    "failure_analysis), random LLM glitches, unfixable environmental constraints. "
    "Be conservative — only propose inquiry when it would CLEARLY help future "
    "attempts. Reply with strict JSON only."
)


def reflect_inquiry_from_failure(
    trajectory: Trajectory,
    goal: Goal,
    verdict: Verdict,
    llm_call: Callable[..., Any] | None = None,
) -> Goal | None:
    """v1.0 #33: при FAIL — определяет, есть ли knowledge-gap, и spawn'ит inquiry.

    Возвращает inquiry Goal (origin=SELF, subtype=LEARNING) или None.

    Fall-open: при ошибках LLM или парсинга возвращает None.

    Использование: вызвать ПАРАЛЛЕЛЬНО с reflect_failure_analysis.
    failure_analysis fix'ит skill template; этот режим расширяет знание.
    """
    if llm_call is None:
        from harnes.llm import call as default_call

        llm_call = default_call

    user_prompt = f"""Failed trajectory analysis for inquiry-worthy knowledge gaps.

Goal: {goal.description}
Goal class: {goal.goal_class.value}

Verdict: {verdict.status.value} (measured_by={verdict.measured_by})
Verdict reasons: {'; '.join(verdict.reasons) if verdict.reasons else '(none)'}

Trajectory:
{_format_trajectory_for_reflect(trajectory)}

Question: is there a CONCRETE knowledge gap (a fact, an API, a concept) that
the agent could investigate as a follow-up inquiry, and that would clearly
improve future attempts on similar goals?

Reply with JSON only:
{{
  "should_spawn_inquiry": true | false,
  "inquiry_description": "<short imperative, e.g. 'Find out how X works'>",
  "rationale": "<one short sentence>"
}}"""

    messages = [
        {"role": "system", "content": _INQUIRY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = llm_call(messages, tier="main", max_tokens=600)
        text = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "reflect.inquiry.llm_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    parsed = _parse_reflect_json(text)
    if parsed is None:
        log.warning("reflect.inquiry.unparseable", raw=text[:300])
        return None

    should_spawn = bool(parsed.get("should_spawn_inquiry", False))
    description = str(parsed.get("inquiry_description", "")).strip()
    rationale = str(parsed.get("rationale", "")).strip()

    log.info(
        "reflect.inquiry.done",
        should_spawn=should_spawn,
        description=description[:120],
        rationale=rationale[:120],
    )

    if not should_spawn or not description:
        return None

    # Простая гигиена: description слишком короткое = не годится.
    if len(description) < 8:
        log.debug("reflect.inquiry.too_short", description=description)
        return None

    inquiry = Goal(
        description=description,
        goal_class=GoalClass.INQUIRY,
        priority=1,  # ниже task priority по умолчанию
        predicate_of_success=JudgePredicate(
            criterion=f"Answer to '{description}' obtained or escalated"
        ),
        origin=Origin.SELF,
        origin_subtype=OriginSubtype.LEARNING,
        originator=f"reflect.inquiry_from_failure:{goal.id}",
        parent_id=goal.id,
        metadata={
            "rationale": rationale,
            "from_trajectory_id": str(trajectory.id),
            "from_verdict_status": verdict.status.value,
        },
    )
    return inquiry
