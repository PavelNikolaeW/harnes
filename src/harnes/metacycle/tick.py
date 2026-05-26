"""Meta-cycle tick driver — 9 этапов как чистые функции над TickState.

См. `agent_architecture.html` § 3.

Стадии:
  sense → attend → goal_arbitration → recall → react_loop
  → verify → world_update → [reflect] → store

В v0:
- attend — простой rule-based scoring (relevance=1, urgency по source-тегу).
- goal_arbitration — picks highest-priority pending goal; декомпозиция отложена.
- react_loop — принимает callable (`react_fn`) извне; так #9 и #10 развязаны.
- verify — только structural-предикат проверяется реально, остальное → UNDETERMINED.
- world_update — stub.
- reflect — SKIPPED.
- store — пишет trajectory в LanceDB + обновляет статус цели по verdict'у.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable, Protocol

import structlog

from harnes.goals.schema import Goal, GoalStatus
from harnes.goals.store import GoalRepository
from harnes.memory.episodic import EpisodicStore
from harnes.memory.router import MemoryRouter
from harnes.memory.schema import MemoryBundle
from harnes.memory.world import WorldModelStore
from harnes.metacycle.schema import (
    FocusFrame,
    ObservationBundle,
    SalientItem,
    SenseObservation,
    Verdict,
    VerifyStatus,
)
from harnes.metacycle.verifiers import verify as _verify_dispatch
from harnes.react.schema import Trajectory, TrajectoryStatus

log = structlog.get_logger()


# ---------- TickState ----------


@dataclass
class TickState:
    """Состояние, прокидываемое через все этапы тика."""

    tick_id: int
    observations: ObservationBundle = field(default_factory=ObservationBundle)
    focus: FocusFrame | None = None
    active_goal: Goal | None = None
    memory: MemoryBundle | None = None
    trajectory: Trajectory | None = None
    verdict: Verdict | None = None
    idle: bool = True


# ---------- React function signature ----------


class ReactFn(Protocol):
    """Контракт стадии react_loop — реализуется в задаче #10."""

    def __call__(
        self,
        active_goal: Goal,
        focus: FocusFrame | None,
        memory: MemoryBundle | None,
    ) -> Trajectory: ...


def stub_react_fn(
    active_goal: Goal,
    focus: FocusFrame | None,
    memory: MemoryBundle | None,
) -> Trajectory:
    """Stub для v0 до подключения настоящего ReAct из #10.

    Возвращает Trajectory со статусом SUCCESS и плейсхолдерным final_state.
    Это позволяет прогонять метацикл end-to-end без LLM-вызовов.
    """
    return Trajectory(
        goal_id=active_goal.id,
        status=TrajectoryStatus.SUCCESS,
        final_state={"stub": True, "goal_description": active_goal.description},
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
    )


# ---------- Stages ----------


def sense(state: TickState, event_queue: list[SenseObservation]) -> TickState:
    """Дренаж event-queue. v0: только push-события из очереди (CLI поднимает их)."""
    if event_queue:
        state.observations.items.extend(event_queue)
        event_queue.clear()
    log.debug("metacycle.sense", tick=state.tick_id, observations=len(state.observations.items))
    return state


def attend(state: TickState) -> TickState:
    """Простое scoring: relevance=1 везде, urgency по source-тегу."""
    items: list[SalientItem] = []
    for obs in state.observations.items:
        urgency = 1.0 if obs.source in ("alert", "operator") else 0.3
        items.append(
            SalientItem(
                observation_id=obs.id,
                relevance=1.0,
                novelty=0.5,
                urgency=urgency,
                score=urgency,  # v0: score = urgency
            )
        )
    state.focus = FocusFrame(
        salient_items=items,
        novelty_score=0.5 if items else 0.0,
        urgency_score=max((i.urgency for i in items), default=0.0),
    )
    log.debug("metacycle.attend", tick=state.tick_id, salient=len(items))
    return state


def goal_arbitration(state: TickState, goal_repo: GoalRepository) -> TickState:
    """Picks highest-priority pending goal. v0: без декомпозиции, без self-целей."""
    pending = goal_repo.list_by_status(GoalStatus.PENDING)
    if not pending:
        state.idle = True
        log.debug("metacycle.goal_arbitration.idle", tick=state.tick_id)
        return state

    active = max(pending, key=lambda g: g.priority)
    active.status = GoalStatus.ACTIVE
    goal_repo.update(active)
    state.active_goal = active
    state.idle = False
    log.info(
        "metacycle.goal_arbitration.active",
        tick=state.tick_id,
        goal_id=str(active.id),
        priority=active.priority,
    )
    return state


def recall_stage(state: TickState, router: MemoryRouter, k: int = 5) -> TickState:
    if state.active_goal is None:
        return state
    state.memory = router.recall(query=state.active_goal.description, k=k)
    log.debug(
        "metacycle.recall",
        tick=state.tick_id,
        episodic=len(state.memory.episodic),
        semantic=len(state.memory.semantic),
    )
    return state


def react_loop_stage(state: TickState, react_fn: ReactFn) -> TickState:
    if state.active_goal is None:
        return state
    state.trajectory = react_fn(
        active_goal=state.active_goal,
        focus=state.focus,
        memory=state.memory,
    )
    log.info(
        "metacycle.react.done",
        tick=state.tick_id,
        trajectory_id=str(state.trajectory.id),
        status=state.trajectory.status,
        steps=len(state.trajectory.steps),
    )
    return state


def verify_stage(state: TickState) -> TickState:
    """Делегирует в harnes.metacycle.verifiers.verify (per-predicate dispatch).

    Поддерживает: structural, judge (LLM-судья), external (deferred).
    State_change и composite в v0.1 — stub UNDETERMINED.
    """
    if state.trajectory is None or state.active_goal is None:
        return state

    state.verdict = _verify_dispatch(state.trajectory, state.active_goal)

    log.info(
        "metacycle.verify",
        tick=state.tick_id,
        verdict=state.verdict.status,
        measured_by=state.verdict.measured_by,
    )
    return state


def world_update_stage(
    state: TickState, world: WorldModelStore | None
) -> TickState:
    """v0 stub: одна запись эпизода через WorldModelStore (он сам в stub-режиме)."""
    if world is None or state.trajectory is None or state.active_goal is None:
        return state

    world.add_episode(
        name=f"trajectory_{state.trajectory.id}",
        episode_body=(
            f"Goal: {state.active_goal.description}. "
            f"Status: {state.trajectory.status}. "
            f"Verdict: {state.verdict.status if state.verdict else 'none'}."
        ),
        source_description="metacycle.world_update",
    )
    return state


def store_stage(
    state: TickState,
    episodic: EpisodicStore,
    goal_repo: GoalRepository,
) -> TickState:
    """Записываем trajectory + обновляем статус цели по verdict'у."""
    if state.trajectory is None or state.active_goal is None:
        return state

    episodic.write_trajectory(state.trajectory)

    if state.verdict is not None:
        if state.verdict.status == VerifyStatus.SUCCESS:
            state.active_goal.status = GoalStatus.DONE
        elif state.verdict.status == VerifyStatus.FAIL:
            state.active_goal.status = GoalStatus.FAILED
        # PARTIAL / UNDETERMINED — оставляем ACTIVE для следующего тика.

        goal_repo.update(state.active_goal)

    log.debug(
        "metacycle.store.done",
        tick=state.tick_id,
        goal_status=state.active_goal.status,
    )
    return state


# ---------- Driver ----------


def run_tick(
    tick_id: int,
    event_queue: list[SenseObservation],
    goal_repo: GoalRepository,
    memory_router: MemoryRouter,
    episodic: EpisodicStore,
    react_fn: ReactFn = stub_react_fn,
    world: WorldModelStore | None = None,
) -> TickState:
    """Один атомарный тик метацикла. Возвращает финальный TickState."""
    state = TickState(tick_id=tick_id)

    state = sense(state, event_queue)
    state = attend(state)
    state = goal_arbitration(state, goal_repo)

    if state.idle:
        log.debug("metacycle.tick.idle", tick=tick_id)
        return state

    state = recall_stage(state, memory_router)
    state = react_loop_stage(state, react_fn)
    state = verify_stage(state)
    state = world_update_stage(state, world)
    # reflect — SKIPPED в v0 (см. § 15, триггерный, не каждый тик)
    state = store_stage(state, episodic, goal_repo)

    log.info("metacycle.tick.done", tick=tick_id, goal_status=state.active_goal.status if state.active_goal else None)
    return state
