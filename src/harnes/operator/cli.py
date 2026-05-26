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


# ---------- bootstrap-standing ----------


@cli.command("bootstrap-standing")
def bootstrap_standing_cmd() -> None:
    """Создать стартовый набор standing-целей (идемпотентно).

    Standing-goals — реактивный слой: постоянно активные политики, которые
    наблюдают состояние и порождают task-подцели при срабатывании.
    """
    from harnes.metacycle.standing import bootstrap_starter_standing_goals

    repo = _open_repo()
    created = bootstrap_starter_standing_goals(repo)
    if not created:
        click.echo("All starter standing-goals already exist.")
        return
    click.echo(f"Created {len(created)} standing-goals:")
    for g in created:
        click.echo(f"  {g.id} :: {g.description}")


# ---------- trace explorer ----------


@cli.command("inspect-trajectory")
@click.argument("trajectory_id")
def inspect_trajectory(trajectory_id: str) -> None:
    """Полный Trajectory из LanceDB: meta + все шаги."""
    from harnes.memory.episodic import EpisodicStore

    settings = get_settings()
    Path(settings.memory.lancedb_path).mkdir(parents=True, exist_ok=True)
    episodic = EpisodicStore(settings.memory.lancedb_path)

    tid = UUID(trajectory_id)
    meta = episodic.get_trajectory_meta(tid)
    if meta is None:
        click.echo(f"Trajectory {trajectory_id} not found", err=True)
        sys.exit(1)

    click.echo("=== Trajectory ===")
    for k, v in meta.items():
        click.echo(f"  {k}: {v}")

    steps = episodic.get_steps(tid)
    click.echo(f"\n=== Steps ({len(steps)}) ===")
    for i, s in enumerate(steps, 1):
        click.echo(
            f"[{i:>3}] {s['step_type']:>12} "
            f"@{s['timestamp']}  "
            f"cost={s['cost_tokens']}t/{s['cost_latency']:.2f}s"
        )
        content = s.get("content_json", "")
        if content:
            try:
                pretty = json.dumps(json.loads(content), indent=4)
                for line in pretty.splitlines():
                    click.echo(f"      {line}")
            except json.JSONDecodeError:
                click.echo(f"      {content}")


@cli.command("recent-trajectories")
@click.option("--limit", type=int, default=20, show_default=True)
@click.option(
    "--status",
    type=click.Choice(["success", "failure", "budget_exceeded", "abandoned"]),
    default=None,
)
def recent_trajectories(limit: int, status: str | None) -> None:
    """Последние N трейекторий (по started_at desc)."""
    from harnes.memory.episodic import EpisodicStore

    settings = get_settings()
    Path(settings.memory.lancedb_path).mkdir(parents=True, exist_ok=True)
    episodic = EpisodicStore(settings.memory.lancedb_path)

    rows = episodic.recent_trajectories(limit=limit, status=status)
    if not rows:
        click.echo("(no trajectories)")
        return

    for r in rows:
        click.echo(
            f"{r['id']} [{r['status']:<16}] "
            f"goal={r['goal_id']} "
            f"started={r['started_at']} "
            f"tokens={r['total_cost_tokens']}"
        )


@cli.command("recent-steps")
@click.option("--limit", type=int, default=30, show_default=True)
def recent_steps(limit: int) -> None:
    """Последние N шагов across all trajectories."""
    from harnes.memory.episodic import EpisodicStore

    settings = get_settings()
    Path(settings.memory.lancedb_path).mkdir(parents=True, exist_ok=True)
    episodic = EpisodicStore(settings.memory.lancedb_path)

    rows = episodic.recent_steps(limit=limit)
    if not rows:
        click.echo("(no steps)")
        return

    for r in rows:
        click.echo(
            f"{r['timestamp']} [{r['step_type']:>12}] "
            f"traj={r['trajectory_id'][:8]}… "
            f"cost={r['cost_tokens']}t"
        )


@cli.command("goal-tree")
@click.argument("goal_id")
def goal_tree(goal_id: str) -> None:
    """ASCII-дерево потомков (рекурсивно по parent_id) + список depends_on."""
    repo = _open_repo()
    root = repo.get(UUID(goal_id))
    if root is None:
        click.echo(f"Goal {goal_id} not found", err=True)
        sys.exit(1)

    def _walk(node: "Goal", prefix: str = "", is_last: bool = True) -> None:  # type: ignore[name-defined]
        marker = "└── " if is_last else "├── "
        click.echo(
            f"{prefix}{marker}{node.id} [{node.status.value:<16}] "
            f"prio={node.priority:>2} :: {node.description}"
        )
        if node.depends_on:
            for dep_id in node.depends_on:
                click.echo(f"{prefix}{'    ' if is_last else '│   '}    depends_on → {dep_id}")
        children = repo.list_children(node.id)
        next_prefix = prefix + ("    " if is_last else "│   ")
        for i, c in enumerate(children):
            _walk(c, next_prefix, i == len(children) - 1)

    _walk(root)


# ---------- run-tick ----------


@cli.command("run-tick")
@click.option(
    "--real",
    is_flag=True,
    default=False,
    help="Использовать настоящий ReAct (дёргает LLM) вместо stub'а",
)
@click.option(
    "--world/--no-world",
    default=True,
    show_default=True,
    help="Подключать ли WorldModelStore (Graphiti+Neo4j) для world_update",
)
def run_tick_cmd(real: bool, world: bool) -> None:
    """Один тик метацикла.

    По умолчанию — stub ReAct (no LLM). С --real — настоящий цикл (LLM-calls).
    World model подключается если --world (default); ошибки Neo4j swallowed.
    """
    from harnes.memory.episodic import EpisodicStore
    from harnes.memory.router import MemoryRouter
    from harnes.metacycle.tick import run_tick, stub_react_fn

    settings = get_settings()
    repo = _open_repo()
    Path(settings.memory.lancedb_path).mkdir(parents=True, exist_ok=True)
    episodic = EpisodicStore(settings.memory.lancedb_path)

    world_store = None
    if world:
        from harnes.memory.world import WorldModelStore

        world_store = WorldModelStore(
            settings.memory.neo4j_uri,
            settings.memory.neo4j_user,
            settings.memory.neo4j_password,
        )

    router = MemoryRouter(episodic=episodic, world=world_store)

    react_fn = stub_react_fn
    skill_registry_for_reflect = None
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
        skill_registry_for_reflect = skill_registry

        def real_react(active_goal, focus, memory):
            return run_react(
                active_goal=active_goal,
                skill=skill_registry.get("general") or general,
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
        world=world_store,
        skill_registry=skill_registry_for_reflect,
    )

    if world_store is not None:
        world_store.close()

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


# ---------- run-loop ----------


@cli.command("run-loop")
@click.option(
    "--interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Секунды между тиками",
)
@click.option(
    "--stub",
    is_flag=True,
    default=False,
    help="Использовать stub ReAct (без LLM) — для smoke-проверки цикла",
)
@click.option(
    "--max-ticks",
    type=int,
    default=None,
    help="Остановиться после N тиков (для dev и тестов). По умолчанию — бесконечно.",
)
@click.option(
    "--world/--no-world",
    default=True,
    show_default=True,
    help="Подключать ли WorldModelStore (Graphiti+Neo4j)",
)
def run_loop(interval: float, stub: bool, max_ticks: int | None, world: bool) -> None:
    """Непрерывный метацикл. Ctrl+C — graceful shutdown.

    На каждом тике: sense → attend → goal_arbitration → (если active goal)
    recall → react → verify → world_update → store. Между тиками — sleep.
    """
    import time

    import structlog

    from harnes.memory.episodic import EpisodicStore
    from harnes.memory.router import MemoryRouter
    from harnes.metacycle.tick import run_tick, stub_react_fn
    from harnes.telemetry import setup_logging

    settings = get_settings()
    setup_logging(settings.logging.level)
    log = structlog.get_logger()

    repo = _open_repo()
    Path(settings.memory.lancedb_path).mkdir(parents=True, exist_ok=True)
    episodic = EpisodicStore(settings.memory.lancedb_path)

    world_store = None
    if world:
        from harnes.memory.world import WorldModelStore

        world_store = WorldModelStore(
            settings.memory.neo4j_uri,
            settings.memory.neo4j_user,
            settings.memory.neo4j_password,
        )

    router = MemoryRouter(episodic=episodic, world=world_store)

    # React function — stub или реальный
    react_fn = stub_react_fn
    skill_registry_for_reflect = None
    if not stub:
        from harnes.react.loop import run_react
        from harnes.skills.store import SkillRegistry
        from harnes.tools.registry import get_registry

        skill_registry = SkillRegistry(
            settings.procedural_store.bundles_dir,
            settings.procedural_store.sqlite_path,
        )
        general = skill_registry.get("general")
        if general is None:
            click.echo("Error: 'general' skill not found", err=True)
            sys.exit(1)
        tool_registry = get_registry()
        skill_registry_for_reflect = skill_registry

        def real_react(active_goal, focus, memory):
            return run_react(
                active_goal=active_goal,
                skill=skill_registry.get("general") or general,
                tool_registry=tool_registry,
                focus=focus,
                memory=memory,
            )

        react_fn = real_react

    log.info(
        "metacycle.loop.start",
        interval=interval,
        stub_mode=stub,
        max_ticks=max_ticks,
    )
    click.echo(
        f"Running metacycle (interval={interval}s, "
        f"react={'stub' if stub else 'real LLM'}, "
        f"max_ticks={max_ticks or '∞'}). Ctrl+C to stop."
    )

    tick_id = 0
    processed = 0
    idle_count = 0
    try:
        while True:
            if max_ticks is not None and tick_id >= max_ticks:
                log.info("metacycle.loop.max_ticks_reached", tick=tick_id)
                break

            state = run_tick(
                tick_id=tick_id,
                event_queue=[],
                goal_repo=repo,
                memory_router=router,
                episodic=episodic,
                react_fn=react_fn,
                world=world_store,
                skill_registry=skill_registry_for_reflect,
            )

            if state.idle:
                idle_count += 1
                log.debug("metacycle.loop.idle_tick", tick=tick_id)
            else:
                processed += 1
                verdict = state.verdict.status.value if state.verdict else "none"
                goal_status = (
                    state.active_goal.status.value if state.active_goal else "none"
                )
                log.info(
                    "metacycle.loop.processed_tick",
                    tick=tick_id,
                    goal_id=str(state.active_goal.id) if state.active_goal else None,
                    verdict=verdict,
                    goal_status=goal_status,
                )

            tick_id += 1

            # Sleep только если max_ticks ещё не достигнут
            if max_ticks is None or tick_id < max_ticks:
                time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nGraceful shutdown — Ctrl+C received.")
    finally:
        if world_store is not None:
            world_store.close()
        log.info(
            "metacycle.loop.stopped",
            total_ticks=tick_id,
            processed=processed,
            idle=idle_count,
        )
        click.echo(
            f"Stopped after {tick_id} ticks ({processed} processed, {idle_count} idle)."
        )


# ---------- entry ----------


if __name__ == "__main__":
    cli()
