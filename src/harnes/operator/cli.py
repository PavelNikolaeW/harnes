"""Operator CLI.

См. `agent_architecture.html` § 6 (operator-approval flow) и § 17.

Команды v0:
- enter-goal      — создаёт цель в PENDING
- list-goals      — список целей (с фильтром по статусу)
- approve / reject — обработка PENDING_APPROVAL
- inspect         — детали по UUID цели
- run-tick        — один тик метацикла (с stub ReAct в v0)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import UUID

import click

from harnes.config import get_settings
from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
    StateChangePredicate,
    StructuralPredicate,
)
from harnes.goals.store import GoalRepository


# ---------- helpers ----------


def _open_repo() -> GoalRepository:
    settings = get_settings()
    return GoalRepository(settings.goal_store.sqlite_path)


def _build_predicate(
    kind: str, criterion: str | None
) -> JudgePredicate | StructuralPredicate | StateChangePredicate:
    if kind == "judge":
        return JudgePredicate(criterion=criterion or "operator-judged complete")
    if kind == "structural":
        return StructuralPredicate(expected_schema={"type": "object"})
    # state_change
    return StateChangePredicate(check_tool_id="", expected_outcome={})


# ---------- root ----------


@click.group()
def cli() -> None:
    """harnes operator CLI."""


# ---------- enter-goal ----------


@cli.command("enter-goal")
@click.argument("description")
@click.option(
    "--class",
    "goal_class",
    type=click.Choice([c.value for c in GoalClass]),
    default=GoalClass.TASK.value,
    show_default=True,
)
@click.option("--priority", type=int, default=0, show_default=True)
@click.option(
    "--predicate",
    type=click.Choice(["judge", "structural", "state_change"]),
    default="judge",
    show_default=True,
)
@click.option("--criterion", default=None, help="Текст критерия (для judge)")
def enter_goal(
    description: str,
    goal_class: str,
    priority: int,
    predicate: str,
    criterion: str | None,
) -> None:
    """Создать новую цель (status=PENDING)."""
    repo = _open_repo()
    goal = Goal(
        description=description,
        goal_class=GoalClass(goal_class),
        priority=priority,
        predicate_of_success=_build_predicate(predicate, criterion),
        origin=Origin.OPERATOR,
        originator="cli",
    )
    repo.create(goal)
    click.echo(f"Created goal {goal.id}")
    click.echo(f"  description: {goal.description}")
    click.echo(f"  class: {goal.goal_class.value}")
    click.echo(f"  priority: {goal.priority}")
    click.echo(f"  status: {goal.status.value}")


# ---------- list-goals ----------


@cli.command("list-goals")
@click.option(
    "--status",
    "status_filter",
    type=click.Choice([s.value for s in GoalStatus]),
    default=None,
)
def list_goals(status_filter: str | None) -> None:
    """Список целей (опционально по статусу)."""
    repo = _open_repo()
    if status_filter is not None:
        goals = repo.list_by_status(GoalStatus(status_filter))
    else:
        goals = []
        for s in GoalStatus:
            goals.extend(repo.list_by_status(s))

    if not goals:
        click.echo("(no goals)")
        return

    for g in goals:
        click.echo(
            f"{g.id} [{g.status.value:18}] prio={g.priority:>2} "
            f"class={g.goal_class.value:11} :: {g.description}"
        )


# ---------- approve / reject ----------


@cli.command("approve")
@click.argument("goal_id")
def approve(goal_id: str) -> None:
    """Approve PENDING_APPROVAL → PENDING."""
    repo = _open_repo()
    try:
        goal = repo.approve(UUID(goal_id))
    except (KeyError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Approved {goal.id} → status={goal.status.value}")


@cli.command("reject")
@click.argument("goal_id")
@click.option("--reason", default="rejected by operator")
def reject(goal_id: str, reason: str) -> None:
    """Reject PENDING_APPROVAL → ABANDONED (с тегом причины в metadata)."""
    repo = _open_repo()
    try:
        goal = repo.reject(UUID(goal_id), reason)
    except (KeyError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Rejected {goal.id} → status={goal.status.value}, reason={reason!r}")


# ---------- inspect ----------


@cli.command("inspect")
@click.argument("goal_id")
def inspect(goal_id: str) -> None:
    """Полный JSON по цели."""
    repo = _open_repo()
    goal = repo.get(UUID(goal_id))
    if goal is None:
        click.echo(f"Goal {goal_id} not found", err=True)
        sys.exit(1)
    click.echo(goal.model_dump_json(indent=2))


# ---------- run-tick ----------


@cli.command("run-tick")
@click.option(
    "--real",
    is_flag=True,
    default=False,
    help="Использовать настоящий ReAct (дёргает LLM) вместо stub'а",
)
def run_tick_cmd(real: bool) -> None:
    """Один тик метацикла.

    По умолчанию — stub ReAct (no LLM). С --real — настоящий цикл (LLM-calls).
    """
    from harnes.memory.episodic import EpisodicStore
    from harnes.memory.router import MemoryRouter
    from harnes.metacycle.tick import run_tick, stub_react_fn

    settings = get_settings()
    repo = _open_repo()
    Path(settings.memory.lancedb_path).mkdir(parents=True, exist_ok=True)
    episodic = EpisodicStore(settings.memory.lancedb_path)
    router = MemoryRouter(episodic=episodic)

    react_fn = stub_react_fn
    if real:
        from harnes.react.loop import run_react
        from harnes.skills.store import SkillRegistry
        from harnes.tools.registry import get_registry

        skill_registry = SkillRegistry(
            settings.procedural_store.bundles_dir,
            settings.procedural_store.sqlite_path,
        )
        general = skill_registry.get("general")
        if general is None:
            click.echo("Error: no 'general' skill found in bundles_dir", err=True)
            sys.exit(1)
        tool_registry = get_registry()

        def real_react(active_goal, focus, memory):
            return run_react(
                active_goal=active_goal,
                skill=general,
                tool_registry=tool_registry,
                focus=focus,
                memory=memory,
            )

        react_fn = real_react

    state = run_tick(
        tick_id=0,
        event_queue=[],
        goal_repo=repo,
        memory_router=router,
        episodic=episodic,
        react_fn=react_fn,
    )

    if state.idle:
        click.echo("Tick idle — no pending goals.")
        return

    assert state.active_goal is not None
    assert state.trajectory is not None
    assert state.verdict is not None
    click.echo("Tick processed:")
    click.echo(f"  goal_id    : {state.active_goal.id}")
    click.echo(f"  trajectory : {state.trajectory.id}")
    click.echo(f"  verdict    : {state.verdict.status.value} (via {state.verdict.measured_by})")
    click.echo(f"  goal status: {state.active_goal.status.value}")


# ---------- entry ----------


if __name__ == "__main__":
    cli()
