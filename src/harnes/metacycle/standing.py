"""Standing-цели — постоянно активный реактивный слой.

См. `agent_architecture.html` § 5.

Standing-goal — это Goal-объект класса STANDING со статусом ACTIVE, который
никогда не закрывается. Его `metadata.policy_name` ссылается на зарегистрированный
здесь callable. На каждом тике (в attend-стадии) policy проверяется; если
условие выстрелило — порождается дочерний task-goal (parent_id = standing.id).

Дедупликация: если у standing-цели уже есть активный/pending child — новый не
создаётся, ждём пока он закроется.

v0.1: 2 стартовые policy:
- `on_alert_observation`     — fires when attend видит наблюдение с urgency≥0.9
- `on_prev_verify_failure`   — fires when в репо есть FAILED цель с приоритетом ≥ threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import structlog

from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
)
from harnes.goals.store import GoalRepository
from harnes.metacycle.schema import FocusFrame

log = structlog.get_logger()


# ---------- Context ----------


@dataclass
class StandingContext:
    """Per-tick контекст для standing policies.

    Заполняется метациклом перед вызовом check_standing_goals().
    """

    tick_id: int
    focus: FocusFrame
    has_active_goal_now: bool


# ---------- Registry ----------


PolicyFn = Callable[[StandingContext, Goal, GoalRepository], Optional[Goal]]
_POLICIES: dict[str, PolicyFn] = {}


def policy(name: str) -> Callable[[PolicyFn], PolicyFn]:
    def deco(fn: PolicyFn) -> PolicyFn:
        if name in _POLICIES:
            raise ValueError(f"Policy {name!r} already registered")
        _POLICIES[name] = fn
        return fn

    return deco


def get_policy(name: str) -> PolicyFn | None:
    return _POLICIES.get(name)


def list_policies() -> list[str]:
    return list(_POLICIES.keys())


# ---------- Starter policies ----------


@policy("on_alert_observation")
def on_alert_observation(
    ctx: StandingContext, parent: Goal, repo: GoalRepository
) -> Goal | None:
    """Fires когда в attend есть наблюдение с urgency >= 0.9 ('alert')."""
    if not any(item.urgency >= 0.9 for item in ctx.focus.salient_items):
        return None
    child_priority = (
        parent.metadata.get("child_priority", 3)
        if isinstance(parent.metadata, dict)
        else 3
    )
    return Goal(
        description=f"Investigate alert observed at tick {ctx.tick_id}",
        goal_class=GoalClass.INQUIRY,
        priority=int(child_priority),
        predicate_of_success=JudgePredicate(
            criterion="alert investigated; root cause or resolution noted"
        ),
        origin=Origin.DECOMPOSITION,
        originator=f"standing:{parent.id}",
        parent_id=parent.id,
    )


@policy("on_prev_verify_failure")
def on_prev_verify_failure(
    ctx: StandingContext, parent: Goal, repo: GoalRepository
) -> Goal | None:
    """Fires когда в репо есть FAILED цель с приоритетом >= threshold.

    Дедуп per-target: для одной и той же исходной FAILED-цели спавнится не более
    `max_diagnoses_per_target` (default=1) diagnose-inquiry — независимо от их
    финального статуса. Иначе уже-диагностированный (но всё ещё FAILED) target
    запускал бесконечный feedback loop на каждом тике после закрытия предыдущего
    diagnose.
    """
    meta = parent.metadata if isinstance(parent.metadata, dict) else {}
    threshold = meta.get("priority_threshold", 1)
    max_per_target = meta.get("max_diagnoses_per_target", 1)

    failed = repo.list_by_status(GoalStatus.FAILED)
    relevant = [g for g in failed if g.priority >= threshold and g.parent_id != parent.id]
    if not relevant:
        return None

    # Подсчёт уже-выпущенных diagnoses по failed_goal_id (в любом статусе).
    diagnose_counts: dict[str, int] = {}
    for child in repo.list_children(parent.id):
        if not isinstance(child.metadata, dict):
            continue
        fgid = child.metadata.get("failed_goal_id")
        if fgid:
            diagnose_counts[fgid] = diagnose_counts.get(fgid, 0) + 1

    relevant = [
        g for g in relevant if diagnose_counts.get(str(g.id), 0) < max_per_target
    ]
    if not relevant:
        return None

    # Самая свежая failed-цель из ещё-не-диагностированных.
    most_recent = max(relevant, key=lambda g: g.updated_at)
    return Goal(
        description=f"Diagnose recent failure: {most_recent.description}",
        goal_class=GoalClass.INQUIRY,
        priority=2,
        predicate_of_success=JudgePredicate(
            criterion="failure root cause identified or explicitly escalated to operator"
        ),
        origin=Origin.DECOMPOSITION,
        originator=f"standing:{parent.id}",
        parent_id=parent.id,
        metadata={"failed_goal_id": str(most_recent.id)},
    )


# ---------- Check + spawn ----------


def _has_active_child(repo: GoalRepository, parent_id) -> bool:
    """Дедупликация — есть ли уже pending/active child у этого standing."""
    children = repo.list_children(parent_id)
    return any(
        c.status in (GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.PENDING_APPROVAL)
        for c in children
    )


def check_standing_goals(
    ctx: StandingContext,
    repo: GoalRepository,
) -> list[Goal]:
    """Проверяет каждую active standing-цель; возвращает список новых spawned task-goals.

    Дедуп: если у standing уже есть pending/active child, новый не создаётся.
    """
    standing_goals = repo.list_by_class(GoalClass.STANDING)
    spawned: list[Goal] = []
    for sg in standing_goals:
        if sg.status != GoalStatus.ACTIVE:
            continue
        if not isinstance(sg.metadata, dict):
            continue
        name = sg.metadata.get("policy_name")
        if not name:
            continue
        fn = get_policy(name)
        if fn is None:
            log.warning(
                "standing.policy.unknown",
                standing_id=str(sg.id),
                policy_name=name,
            )
            continue
        if _has_active_child(repo, sg.id):
            continue

        try:
            new_goal = fn(ctx, sg, repo)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "standing.policy.crashed",
                policy=name,
                standing_id=str(sg.id),
                error=str(exc),
            )
            continue

        if new_goal is not None:
            repo.create(new_goal)
            spawned.append(new_goal)
            log.info(
                "standing.spawned",
                parent_id=str(sg.id),
                policy=name,
                new_goal_id=str(new_goal.id),
            )
    return spawned


# ---------- Bootstrap ----------


_STARTER_STANDING_GOALS = [
    {
        "description": "Investigate environmental alerts (urgency≥0.9)",
        "policy_name": "on_alert_observation",
        "child_priority": 3,
    },
    {
        "description": "Diagnose failures on goals with priority≥2",
        "policy_name": "on_prev_verify_failure",
        "priority_threshold": 2,
    },
]


def bootstrap_starter_standing_goals(repo: GoalRepository) -> list[Goal]:
    """Создаёт стартовый набор standing-целей. Идемпотентно — пропускает
    уже существующие policy_name."""
    existing = {
        g.metadata.get("policy_name")
        for g in repo.list_by_class(GoalClass.STANDING)
        if isinstance(g.metadata, dict)
    }
    created: list[Goal] = []
    for cfg in _STARTER_STANDING_GOALS:
        if cfg["policy_name"] in existing:
            continue
        goal = Goal(
            description=str(cfg["description"]),
            goal_class=GoalClass.STANDING,
            status=GoalStatus.ACTIVE,
            predicate_of_success=JudgePredicate(criterion="never (standing policy)"),
            origin=Origin.OPERATOR,
            originator="bootstrap",
            metadata={k: v for k, v in cfg.items() if k != "description"},
        )
        repo.create(goal)
        created.append(goal)
    return created
