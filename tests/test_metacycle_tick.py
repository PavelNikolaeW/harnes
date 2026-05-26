"""Tests for harnes.metacycle.tick — отдельные стадии и full-tick прогон."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    Origin,
    StructuralPredicate,
)
from harnes.goals.store import GoalRepository
from harnes.memory.episodic import EpisodicStore
from harnes.memory.router import MemoryRouter
from harnes.metacycle.schema import (
    ObservationBundle,
    SalientItem,
    SenseObservation,
    VerifyStatus,
)
from harnes.metacycle.tick import (
    TickState,
    attend,
    goal_arbitration,
    react_loop_stage,
    recall_stage,
    run_tick,
    sense,
    store_stage,
    stub_react_fn,
    verify_stage,
    world_update_stage,
)
from harnes.react.schema import Trajectory, TrajectoryStatus


# ---------- Fixtures ----------


@pytest.fixture
def goal_repo() -> GoalRepository:
    return GoalRepository(":memory:")


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "ep")


@pytest.fixture
def empty_router() -> MemoryRouter:
    return MemoryRouter()


def _make_goal(
    priority: int = 0, status: GoalStatus = GoalStatus.PENDING
) -> Goal:
    return Goal(
        description="write hello to file",
        goal_class=GoalClass.TASK,
        predicate_of_success=StructuralPredicate(expected_schema={"type": "object"}),
        priority=priority,
        status=status,
        origin=Origin.OPERATOR,
        originator="test",
    )


# ---------- Individual stages ----------


def test_sense_drains_event_queue() -> None:
    state = TickState(tick_id=0)
    queue = [
        SenseObservation(source="cli", payload={"goal": "x"}),
        SenseObservation(source="alert", payload={"text": "y"}),
    ]
    sense(state, queue)
    assert len(state.observations.items) == 2
    assert queue == []  # очередь дренирована


def test_attend_marks_urgency_from_source() -> None:
    state = TickState(tick_id=0)
    state.observations.items.append(SenseObservation(source="alert", payload={}))
    state.observations.items.append(SenseObservation(source="timer", payload={}))
    attend(state)
    assert state.focus is not None
    urgencies = {i.urgency for i in state.focus.salient_items}
    assert 1.0 in urgencies  # alert
    assert 0.3 in urgencies  # timer


def test_goal_arbitration_picks_highest_priority(
    goal_repo: GoalRepository,
) -> None:
    low = _make_goal(priority=1)
    high = _make_goal(priority=10)
    goal_repo.create(low)
    goal_repo.create(high)

    state = TickState(tick_id=0)
    goal_arbitration(state, goal_repo)

    assert state.active_goal is not None
    assert state.active_goal.id == high.id
    assert state.active_goal.status == GoalStatus.ACTIVE
    # Persisted as ACTIVE
    refetched = goal_repo.get(high.id)
    assert refetched is not None
    assert refetched.status == GoalStatus.ACTIVE


def test_goal_arbitration_idle_when_no_pending(
    goal_repo: GoalRepository,
) -> None:
    state = TickState(tick_id=0)
    goal_arbitration(state, goal_repo)
    assert state.idle is True
    assert state.active_goal is None


def test_recall_stage_populates_bundle(
    goal_repo: GoalRepository, empty_router: MemoryRouter
) -> None:
    goal = _make_goal()
    goal_repo.create(goal)
    state = TickState(tick_id=0, active_goal=goal)
    recall_stage(state, empty_router)
    assert state.memory is not None  # пусть и пустой


def test_react_loop_stage_uses_provided_fn() -> None:
    goal = _make_goal()
    state = TickState(tick_id=0, active_goal=goal)
    react_loop_stage(state, stub_react_fn)
    assert state.trajectory is not None
    assert state.trajectory.status == TrajectoryStatus.SUCCESS
    assert state.trajectory.final_state is not None


def test_verify_stage_structural_success() -> None:
    goal = _make_goal()
    traj = Trajectory(
        goal_id=goal.id,
        status=TrajectoryStatus.SUCCESS,
        final_state={"ok": True},
    )
    state = TickState(tick_id=0, active_goal=goal, trajectory=traj)
    verify_stage(state)
    assert state.verdict is not None
    assert state.verdict.status == VerifyStatus.SUCCESS
    assert state.verdict.measured_by == "structural"


def test_verify_stage_structural_fail_when_no_final_state() -> None:
    goal = _make_goal()
    traj = Trajectory(goal_id=goal.id, status=TrajectoryStatus.SUCCESS, final_state=None)
    state = TickState(tick_id=0, active_goal=goal, trajectory=traj)
    verify_stage(state)
    assert state.verdict is not None
    assert state.verdict.status == VerifyStatus.FAIL


def test_world_update_stub_does_not_crash() -> None:
    from harnes.memory.world import WorldModelStore

    goal = _make_goal()
    traj = Trajectory(goal_id=goal.id, status=TrajectoryStatus.SUCCESS, final_state={"x": 1})
    state = TickState(tick_id=0, active_goal=goal, trajectory=traj)
    world = WorldModelStore("bolt://nowhere:7687", "u", "p")
    world_update_stage(state, world)
    # No crash = pass


def test_store_stage_writes_and_updates_goal(
    goal_repo: GoalRepository, episodic: EpisodicStore
) -> None:
    from harnes.metacycle.schema import Verdict

    goal = _make_goal()
    goal_repo.create(goal)

    traj = Trajectory(
        goal_id=goal.id,
        status=TrajectoryStatus.SUCCESS,
        final_state={"ok": True},
    )
    state = TickState(
        tick_id=0,
        active_goal=goal,
        trajectory=traj,
        verdict=Verdict(status=VerifyStatus.SUCCESS, measured_by="structural"),
    )
    store_stage(state, episodic, goal_repo)

    # Goal status обновлён
    refetched = goal_repo.get(goal.id)
    assert refetched is not None
    assert refetched.status == GoalStatus.DONE

    # Trajectory сохранена в episodic
    meta = episodic.get_trajectory_meta(traj.id)
    assert meta is not None


# ---------- Full-tick integration ----------


def test_full_tick_idle(
    goal_repo: GoalRepository,
    episodic: EpisodicStore,
    empty_router: MemoryRouter,
) -> None:
    """Без целей и событий — тик idle."""
    state = run_tick(
        tick_id=0,
        event_queue=[],
        goal_repo=goal_repo,
        memory_router=empty_router,
        episodic=episodic,
    )
    assert state.idle is True
    assert state.trajectory is None


def test_full_tick_completes_goal(
    goal_repo: GoalRepository,
    episodic: EpisodicStore,
    empty_router: MemoryRouter,
) -> None:
    """Полный тик: pending goal → ACTIVE → ReAct (stub) → verify → store → DONE."""
    goal = _make_goal(priority=5)
    goal_repo.create(goal)

    state = run_tick(
        tick_id=0,
        event_queue=[],
        goal_repo=goal_repo,
        memory_router=empty_router,
        episodic=episodic,
    )

    assert state.idle is False
    assert state.trajectory is not None
    assert state.verdict is not None
    assert state.verdict.status == VerifyStatus.SUCCESS

    # Goal в финальном статусе
    refetched = goal_repo.get(goal.id)
    assert refetched is not None
    assert refetched.status == GoalStatus.DONE

    # Trajectory в episodic
    meta = episodic.get_trajectory_meta(state.trajectory.id)
    assert meta is not None
    assert meta["status"] == "success"


def test_full_tick_with_failing_react(
    goal_repo: GoalRepository,
    episodic: EpisodicStore,
    empty_router: MemoryRouter,
) -> None:
    """ReAct, возвращающий final_state=None → verify=FAIL → goal=FAILED."""
    goal = _make_goal()
    goal_repo.create(goal)

    def failing_react(active_goal, focus, memory):
        return Trajectory(
            goal_id=active_goal.id,
            status=TrajectoryStatus.FAILURE,
            final_state=None,
        )

    state = run_tick(
        tick_id=0,
        event_queue=[],
        goal_repo=goal_repo,
        memory_router=empty_router,
        episodic=episodic,
        react_fn=failing_react,
    )

    assert state.verdict.status == VerifyStatus.FAIL
    refetched = goal_repo.get(goal.id)
    assert refetched.status == GoalStatus.FAILED
