"""Tests for v1.0 #33: Self-generated goals.

Coverage:
- reflect.reflect_inquiry_from_failure → Goal или None
- run_tick собирает spawned_goals (standing + reflect inquiry)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
    OriginSubtype,
)
from harnes.goals.store import GoalRepository
from harnes.memory.episodic import EpisodicStore
from harnes.memory.router import MemoryRouter
from harnes.metacycle.reflect import reflect_inquiry_from_failure
from harnes.metacycle.schema import (
    SenseObservation,
    Verdict,
    VerifyStatus,
)
from harnes.metacycle.tick import TickState, run_tick, stub_react_fn
from harnes.react.schema import (
    ActionStep,
    ObservationOutcome,
    ObservationStep,
    ThoughtStep,
    Trajectory,
    TrajectoryStatus,
)


# ---------- Fixtures ----------


@pytest.fixture
def goal() -> Goal:
    return Goal(
        description="Find the population of Atlantis in 2020",
        goal_class=GoalClass.TASK,
        predicate_of_success=JudgePredicate(criterion="population reported with source"),
        origin=Origin.OPERATOR,
        originator="test",
    )


@pytest.fixture
def failing_trajectory(goal: Goal) -> Trajectory:
    return Trajectory(
        goal_id=goal.id,
        steps=[
            ThoughtStep(text="I don't know what Atlantis is"),
            ActionStep(tool_id="search", args={"query": "Atlantis population"}),
            ObservationStep(
                outcome=ObservationOutcome.TOOL_ERROR,
                error_detail="tool 'search' not in registry",
            ),
        ],
        status=TrajectoryStatus.FAILURE,
        final_state=None,
    )


@pytest.fixture
def fail_verdict() -> Verdict:
    return Verdict(
        status=VerifyStatus.FAIL,
        reasons=["agent lacked knowledge of Atlantis"],
        measured_by="judge_llm",
    )


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=50, completion_tokens=100)
    return response


# ---------- reflect_inquiry_from_failure ----------


def test_inquiry_spawned_when_should_spawn_true(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
) -> None:
    payload = {
        "should_spawn_inquiry": True,
        "inquiry_description": "Find out what 'Atlantis' refers to in the user's domain",
        "rationale": "The agent did not recognise the term, blocking the goal.",
    }
    fake_llm = MagicMock(return_value=_mock_response(json.dumps(payload)))

    inquiry = reflect_inquiry_from_failure(
        trajectory=failing_trajectory,
        goal=goal,
        verdict=fail_verdict,
        llm_call=fake_llm,
    )

    assert inquiry is not None
    assert inquiry.goal_class == GoalClass.INQUIRY
    assert inquiry.origin == Origin.SELF
    assert inquiry.origin_subtype == OriginSubtype.LEARNING
    assert inquiry.parent_id == goal.id
    assert "Atlantis" in inquiry.description
    assert inquiry.metadata["rationale"].startswith("The agent")
    assert inquiry.metadata["from_trajectory_id"] == str(failing_trajectory.id)

    # LLM called с tier=main
    assert fake_llm.call_args.kwargs["tier"] == "main"


def test_inquiry_none_when_should_spawn_false(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
) -> None:
    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"should_spawn_inquiry": false, "inquiry_description": "", "rationale": "environment"}'
        )
    )

    inquiry = reflect_inquiry_from_failure(
        trajectory=failing_trajectory,
        goal=goal,
        verdict=fail_verdict,
        llm_call=fake_llm,
    )
    assert inquiry is None


def test_inquiry_none_on_empty_description(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
) -> None:
    """should_spawn=true но description пустой → None."""
    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"should_spawn_inquiry": true, "inquiry_description": "", "rationale": "x"}'
        )
    )
    assert (
        reflect_inquiry_from_failure(
            failing_trajectory, goal, fail_verdict, llm_call=fake_llm
        )
        is None
    )


def test_inquiry_none_on_too_short_description(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
) -> None:
    """description < 8 символов отбрасывается."""
    fake_llm = MagicMock(
        return_value=_mock_response(
            '{"should_spawn_inquiry": true, "inquiry_description": "huh?", "rationale": "x"}'
        )
    )
    assert (
        reflect_inquiry_from_failure(
            failing_trajectory, goal, fail_verdict, llm_call=fake_llm
        )
        is None
    )


def test_inquiry_none_on_unparseable_llm_output(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
) -> None:
    fake_llm = MagicMock(return_value=_mock_response("not JSON at all"))
    assert (
        reflect_inquiry_from_failure(
            failing_trajectory, goal, fail_verdict, llm_call=fake_llm
        )
        is None
    )


def test_inquiry_none_on_llm_exception(
    goal: Goal,
    failing_trajectory: Trajectory,
    fail_verdict: Verdict,
) -> None:
    fake_llm = MagicMock(side_effect=RuntimeError("llm down"))
    # Не должен пробрасываться — graceful fall-open.
    assert (
        reflect_inquiry_from_failure(
            failing_trajectory, goal, fail_verdict, llm_call=fake_llm
        )
        is None
    )


# ---------- run_tick собирает spawned_goals ----------


def _trivial_router(tmp_path: Path) -> tuple[MemoryRouter, EpisodicStore]:
    episodic = EpisodicStore(tmp_path / "ep")
    router = MemoryRouter(episodic=episodic)
    return router, episodic


def test_tick_state_has_spawned_goals_field() -> None:
    state = TickState(tick_id=0)
    assert state.spawned_goals == []


def test_run_tick_aggregates_standing_spawned_goals(tmp_path: Path) -> None:
    """Standing-policy создаёт child-goal → попадает в state.spawned_goals."""
    from harnes.metacycle.standing import bootstrap_starter_standing_goals

    repo = GoalRepository(tmp_path / "g.db")
    bootstrap_starter_standing_goals(repo)
    router, episodic = _trivial_router(tmp_path)

    # Создаём провалившуюся цель с priority=2 — это активирует on_prev_verify_failure.
    failed_goal = Goal(
        description="some failed task",
        goal_class=GoalClass.TASK,
        priority=2,
        predicate_of_success=JudgePredicate(criterion="x"),
        origin=Origin.OPERATOR,
        originator="test",
    )
    failed_goal.status = GoalStatus.FAILED
    repo.create(failed_goal)

    state = run_tick(
        tick_id=1,
        event_queue=[],
        goal_repo=repo,
        memory_router=router,
        episodic=episodic,
        react_fn=stub_react_fn,
    )

    # У нас должен появиться spawn'нутый INQUIRY-goal
    assert len(state.spawned_goals) >= 1
    spawned_descriptions = [g.description for g in state.spawned_goals]
    assert any("failed task" in d or "Diagnose" in d for d in spawned_descriptions)


def test_run_tick_spawns_inquiry_on_fail_verdict(tmp_path: Path, monkeypatch) -> None:
    """При FAIL verify reflect.inquiry_from_failure spawn'ит inquiry в repo и state."""
    from harnes.metacycle import reflect as reflect_mod
    from harnes.metacycle import tick as tick_mod

    repo = GoalRepository(tmp_path / "g.db")
    router, episodic = _trivial_router(tmp_path)

    # Создаём task, которая выполнится stub_react_fn и потом провалится на verify.
    task_goal = Goal(
        description="impossible task",
        goal_class=GoalClass.TASK,
        priority=5,
        predicate_of_success=JudgePredicate(criterion="impossible criterion"),
        origin=Origin.OPERATOR,
        originator="test",
    )
    repo.create(task_goal)

    # Подменяем verify-стейдж: всегда возвращает FAIL.
    def fail_verify(state, goal_repo=None):
        if state.trajectory is None or state.active_goal is None:
            return state
        state.verdict = Verdict(
            status=VerifyStatus.FAIL,
            reasons=["forced fail"],
            measured_by="test",
        )
        return state

    monkeypatch.setattr(tick_mod, "verify_stage", fail_verify)

    # Подменяем reflect_inquiry_from_failure — без LLM, возвращает заранее заданный inquiry.
    def fake_inquiry(trajectory, goal, verdict, llm_call=None):
        return Goal(
            description="Find out why impossible task fails",
            goal_class=GoalClass.INQUIRY,
            predicate_of_success=JudgePredicate(criterion="answer found"),
            origin=Origin.SELF,
            origin_subtype=OriginSubtype.LEARNING,
            originator="test_reflect",
            parent_id=goal.id,
        )

    monkeypatch.setattr(reflect_mod, "reflect_inquiry_from_failure", fake_inquiry)

    state = run_tick(
        tick_id=1,
        event_queue=[],
        goal_repo=repo,
        memory_router=router,
        episodic=episodic,
        react_fn=stub_react_fn,
        check_standing=False,
    )

    # inquiry в state.spawned_goals
    assert len(state.spawned_goals) == 1
    spawned = state.spawned_goals[0]
    assert spawned.goal_class == GoalClass.INQUIRY
    assert spawned.origin == Origin.SELF
    assert spawned.origin_subtype == OriginSubtype.LEARNING

    # И главное — он также сохранён в репо.
    fetched = repo.get(spawned.id)
    assert fetched is not None
    assert fetched.description == "Find out why impossible task fails"
    assert fetched.parent_id == task_goal.id


def test_run_tick_inquiry_failure_does_not_crash_tick(
    tmp_path: Path, monkeypatch
) -> None:
    """Если reflect_inquiry бросает — тик завершается чисто."""
    from harnes.metacycle import reflect as reflect_mod
    from harnes.metacycle import tick as tick_mod

    repo = GoalRepository(tmp_path / "g.db")
    router, episodic = _trivial_router(tmp_path)

    task_goal = Goal(
        description="t",
        goal_class=GoalClass.TASK,
        priority=5,
        predicate_of_success=JudgePredicate(criterion="c"),
        origin=Origin.OPERATOR,
        originator="test",
    )
    repo.create(task_goal)

    def fail_verify(state, goal_repo=None):
        if state.trajectory is None or state.active_goal is None:
            return state
        state.verdict = Verdict(
            status=VerifyStatus.FAIL,
            reasons=["forced"],
            measured_by="test",
        )
        return state

    def broken_inquiry(trajectory, goal, verdict, llm_call=None):
        raise RuntimeError("inquiry exploded")

    monkeypatch.setattr(tick_mod, "verify_stage", fail_verify)
    monkeypatch.setattr(reflect_mod, "reflect_inquiry_from_failure", broken_inquiry)

    state = run_tick(
        tick_id=1,
        event_queue=[],
        goal_repo=repo,
        memory_router=router,
        episodic=episodic,
        react_fn=stub_react_fn,
        check_standing=False,
    )

    # Тик не упал. И ничего не spawn'нулось.
    assert state.spawned_goals == []
