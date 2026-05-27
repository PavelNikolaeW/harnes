"""ReAct internal loop — Classic вариант.

См. `agent_architecture.html` § 7.

В v0:
- 2 отдельных модельных вызова за итерацию: thought_call, action_call
- Без критика
- Без plan-шага
- Без subagent
- Loop-detector — n-gram над последовательностью (tool_id, args)
- Бюджет по токенам и по step-count
- Завершение: action с tool_id="finish" → success + final_state из args

Контракты входа/выхода — см. harnes.metacycle.tick.ReactFn.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Callable

import structlog

from harnes.goals.schema import Goal
from harnes.memory.schema import MemoryBundle
from harnes.metacycle.schema import FocusFrame
from harnes.react.schema import (
    ActionStep,
    Cost,
    ObservationOutcome,
    ObservationStep,
    ThoughtStep,
    Trajectory,
    TrajectoryStatus,
)
from harnes.skills.schema import Skill
from harnes.tools.registry import ToolRegistry

log = structlog.get_logger()


# Специальный tool_id, сигнализирующий о завершении траектории.
FINISH_TOOL_ID = "finish"


# ---------- Prompts ----------


def _build_system_prompt(
    skill: Skill,
    goal: Goal,
    tool_registry: ToolRegistry,
    memory: MemoryBundle | None,
) -> str:
    """Конструирует system-prompt из skill template + описаний доступных тулов."""
    tools_section_lines: list[str] = []
    for tool_id in skill.allowed_tools:
        tool = tool_registry.get(tool_id)
        if tool is None:
            continue
        tools_section_lines.append(f"- {tool.id}: {tool.description}")
        tools_section_lines.append(
            f"  args schema: {json.dumps(tool.input_schema.get('properties', {}))}"
        )
    tools_section_lines.append(
        f"- {FINISH_TOOL_ID}: signal task completion."
    )
    tools_section_lines.append(
        f'  args: {{"final_state": <object describing the result>}}'
    )
    tools_section = "\n".join(tools_section_lines)

    memory_section = ""
    if memory is not None and (memory.episodic or memory.semantic):
        memory_section = "\nRelevant memory:\n"
        if memory.semantic:
            memory_section += "Semantic facts:\n"
            for rec in memory.semantic[:5]:
                memory_section += f"- {rec.text}\n"

    # skill.prompt_template поддерживает простые {goal_description} / {tools_list}
    base = skill.prompt_template.format(
        goal_description=goal.description,
        tools_list=tools_section,
    )
    return f"{base}\n{memory_section}\n\nYou act in a strict Thought → Action loop. When done, use tool_id={FINISH_TOOL_ID}."


def _trajectory_as_messages(traj: Trajectory) -> list[dict[str, Any]]:
    """Сериализует пройденные шаги Trajectory в чатовый формат для LLM."""
    messages: list[dict[str, Any]] = []
    for step in traj.steps:
        if step.type == "thought":
            messages.append({"role": "assistant", "content": f"Thought: {step.text}"})
        elif step.type == "action":
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Action: {json.dumps({'tool_id': step.tool_id, 'args': step.args})}",
                }
            )
        elif step.type == "observation":
            payload_str = (
                json.dumps(step.payload) if step.payload is not None else step.error_detail
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation [{step.outcome.value}]: {payload_str}",
                }
            )
    return messages


# ---------- LLM call wrappers ----------


def _thought_call(
    system_prompt: str,
    traj: Trajectory,
    llm_call: Callable[..., Any],
) -> ThoughtStep:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_trajectory_as_messages(traj))
    messages.append(
        {
            "role": "user",
            "content": "What is your next Thought? Reason briefly about what to do next.",
        }
    )

    # max_tokens=2000: native thinking может потратить 500-1500 на reasoning_content,
    # остальное — на content. См. llm/client.py:_thinking_extra_body.
    response = llm_call(messages, max_tokens=2000)
    text = response.choices[0].message.content or ""
    tokens = getattr(response.usage, "completion_tokens", 0) or 0

    # Часто модель пишет "Thought: ..." — нормализуем.
    if text.lower().startswith("thought:"):
        text = text.split(":", 1)[1].strip()

    return ThoughtStep(text=text, cost=Cost(tokens=tokens))


def _action_call(
    system_prompt: str,
    traj: Trajectory,
    llm_call: Callable[..., Any],
) -> ActionStep | None:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_trajectory_as_messages(traj))
    messages.append(
        {
            "role": "user",
            "content": (
                "What is your next Action? "
                'Reply with ONE JSON object only, no other text: '
                '{"tool_id": "name", "args": {...}}. '
                f'To finish, use tool_id={FINISH_TOOL_ID!r} with args={{"final_state": ...}}.'
            ),
        }
    )

    # max_tokens=1500: thinking reasoning_content + JSON ответ в content.
    # Empirical: content остаётся чистым JSON, _parse_action_json regex не цепляет шум.
    response = llm_call(messages, max_tokens=1500)
    text = response.choices[0].message.content or ""
    tokens = getattr(response.usage, "completion_tokens", 0) or 0

    parsed = _parse_action_json(text)
    if parsed is None or "tool_id" not in parsed:
        log.warning("react.action.parse_failed", raw=text[:200])
        return None

    return ActionStep(
        tool_id=str(parsed["tool_id"]),
        args=dict(parsed.get("args", {})),
        cost=Cost(tokens=tokens),
    )


def _parse_action_json(text: str) -> dict[str, Any] | None:
    """Достаёт первый JSON-объект из ответа модели. Любим тех, кто пишет чисто;
    но толерируем «Action: {…}» обвязки."""
    text = text.strip()
    # Попытка 1: прямой parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Попытка 2: выдрать {...} regexp'ом (greedy для вложенных).
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


# ---------- Loop detection ----------


def _detect_loop(traj: Trajectory, window: int = 4) -> bool:
    """Простая детекция: последние `window` action-шагов идентичны
    предыдущим `window` action-шагам.
    """
    actions = [s for s in traj.steps if s.type == "action"]
    if len(actions) < 2 * window:
        return False
    last = actions[-window:]
    prev = actions[-2 * window : -window]
    return all(
        a.tool_id == b.tool_id and a.args == b.args for a, b in zip(last, prev)
    )


# ---------- Main loop ----------


def run_react(
    active_goal: Goal,
    skill: Skill,
    tool_registry: ToolRegistry,
    focus: FocusFrame | None = None,
    memory: MemoryBundle | None = None,
    llm_call: Callable[..., Any] | None = None,
    max_steps: int = 20,
    budget_tokens: int = 50_000,
    with_critic: bool = True,
    max_critic_rejections: int = 3,
) -> Trajectory:
    """Classic ReAct loop с опциональным критиком на irreversible-действиях.

    См. § 7 и § 11.

    Параметры:
    - llm_call: функция (messages, **kwargs) → response. None → harnes.llm.call.
    - with_critic: включает критика перед action с is_irreversible=true.
    - max_critic_rejections: после стольких REJECT'ов подряд траектория FAILURE.
    """
    if llm_call is None:
        from harnes.llm import call as default_call

        llm_call = default_call

    system_prompt = _build_system_prompt(skill, active_goal, tool_registry, memory)

    traj = Trajectory(
        goal_id=active_goal.id,
        started_at=datetime.now(UTC),
        metadata={
            "skill_id": skill.id,
            "skill_version": skill.version,
        },
    )
    total_tokens = 0
    critic_rejections = 0

    for step_idx in range(max_steps):
        # Бюджет
        if total_tokens >= budget_tokens:
            traj.status = TrajectoryStatus.BUDGET_EXCEEDED
            log.info(
                "react.terminated.budget",
                trajectory_id=str(traj.id),
                total_tokens=total_tokens,
            )
            break

        # 1. Thought
        try:
            thought = _thought_call(system_prompt, traj, llm_call)
        except Exception as exc:  # noqa: BLE001
            log.error("react.thought_call.failed", error=str(exc))
            traj.status = TrajectoryStatus.FAILURE
            break
        traj.steps.append(thought)
        total_tokens += thought.cost.tokens

        # 2. Action
        try:
            action = _action_call(system_prompt, traj, llm_call)
        except Exception as exc:  # noqa: BLE001
            log.error("react.action_call.failed", error=str(exc))
            traj.status = TrajectoryStatus.FAILURE
            break

        if action is None:
            # Не смогли распарсить — добавим observation как malformed и продолжим.
            traj.steps.append(
                ObservationStep(
                    outcome=ObservationOutcome.MALFORMED_OUTPUT,
                    error_detail="failed to parse action JSON from model output",
                )
            )
            continue

        # Resolve irreversibility (informational в v0 без критика).
        action.is_irreversible = tool_registry.resolve_irreversibility(
            action.tool_id, action.args, skill=skill
        )
        traj.steps.append(action)
        total_tokens += action.cost.tokens

        # 3. Завершение по tool_id="finish"
        if action.tool_id == FINISH_TOOL_ID:
            traj.status = TrajectoryStatus.SUCCESS
            traj.final_state = action.args.get("final_state")
            log.info(
                "react.terminated.finish",
                trajectory_id=str(traj.id),
                steps=len(traj.steps),
            )
            break

        # 4. Критик на irreversible-действиях (если включён)
        if with_critic and action.is_irreversible:
            from harnes.react.critic import critique_action

            critique = critique_action(action, traj, active_goal, llm_call=llm_call)
            traj.steps.append(critique)
            total_tokens += critique.cost.tokens

            if critique.verdict.value == "reject":
                critic_rejections += 1
                log.info(
                    "react.critic.rejected_action",
                    trajectory_id=str(traj.id),
                    tool_id=action.tool_id,
                    rejection=critic_rejections,
                    reasoning=critique.reasoning[:200],
                )
                if critic_rejections > max_critic_rejections:
                    traj.status = TrajectoryStatus.FAILURE
                    log.warning(
                        "react.terminated.critic_rejections_exceeded",
                        trajectory_id=str(traj.id),
                        limit=max_critic_rejections,
                    )
                    break
                # Skip execution, back to next thought (critic в контексте).
                continue
            # ok / warning — action исполняется.

        # 5. Execute
        obs = tool_registry.invoke(action.tool_id, action.args, skill=skill)
        traj.steps.append(obs)

        # 6. Loop detection
        if _detect_loop(traj):
            log.warning(
                "react.terminated.loop_detected",
                trajectory_id=str(traj.id),
                steps=len(traj.steps),
            )
            traj.status = TrajectoryStatus.FAILURE
            break
    else:
        # Hit max_steps без явного завершения.
        traj.status = TrajectoryStatus.FAILURE
        log.warning(
            "react.terminated.max_steps",
            trajectory_id=str(traj.id),
            max_steps=max_steps,
        )

    traj.ended_at = datetime.now(UTC)
    traj.total_cost = Cost(
        tokens=total_tokens,
        latency_seconds=(traj.ended_at - traj.started_at).total_seconds(),
    )
    return traj
