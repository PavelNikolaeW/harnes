"""Tests for harnes.metacycle.verifiers.verify_composite."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from harnes.goals.schema import (
    Aggregation,
    CompositePredicate,
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
    StructuralPredicate,
)
from harnes.goals.store import GoalRepository
from harnes.metacycle.schema import VerifyStatus
from harnes.metacycle.verifiers import verify_composite
from harnes.react.schema import Trajectory


def _parent_goal(aggregation: Aggregation = Aggregation.ALL, custom_check: str = "") -> Goal:
    return Goal(
        description="parent composite goal",
        goal_class=GoalClass.TASK,
        predicate_of_success=CompositePredicate(
            aggregation=aggregation, custom_check=custom_check or None
        ),
        origin=Origin.OPERATOR,
        originator="test",
    )


def _child(parent_id, status: GoalStatus, description: str = "child") -> Goal:
    return Goal(
        description=description,
        goal_class=GoalClass.TASK,
        predicate_of_success=StructuralPredicate(expected_schema={"type": "object"}),
        status=status,
        origin=Origin.DECOMPOSITION,
        originator="test",
        parent_id=parent_id,
    )


def _traj(goal_id) -> Trajectory:
    return Trajectory(goal_id=goal_id)


# ---------- guard rails ----------


def test_composite_without_goal_repo_undetermined() -> None:
    parent = _parent_goal()
    v = verify_composite(parent.predicate_of_success, _traj(parent.id), parent)
    assert v.status == VerifyStatus.UNDETERMINED
    assert "needs goal_repo" in v.reasons[0]


def test_composite_no_children_undetermined() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal()
    repo.create(parent)
    v = verify_composite(
        parent.predicate_of_success, _traj(parent.id), parent, goal_repo=repo
    )
    assert v.status == VerifyStatus.UNDETERMINED


def test_composite_waits_for_non_terminal_children() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal()
    repo.create(parent)
    repo.create(_child(parent.id, status=GoalStatus.DONE))
    repo.create(_child(parent.id, status=GoalStatus.ACTIVE))  # ещё работает

    v = verify_composite(
        parent.predicate_of_success, _traj(parent.id), parent, goal_repo=repo
    )
    assert v.status == VerifyStatus.UNDETERMINED
    assert "waits" in v.reasons[0]
    assert v.evidence[0]["non_terminal_count"] == 1


# ---------- aggregation=ALL ----------


def test_composite_all_done_success() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal(Aggregation.ALL)
    repo.create(parent)
    repo.create(_child(parent.id, GoalStatus.DONE))
    repo.create(_child(parent.id, GoalStatus.DONE))
    repo.create(_child(parent.id, GoalStatus.DONE))

    v = verify_composite(
        parent.predicate_of_success, _traj(parent.id), parent, goal_repo=repo
    )
    assert v.status == VerifyStatus.SUCCESS
    assert v.measured_by == "composite"
    assert "3/3 children done" in v.reasons[0]


def test_composite_all_one_failed_overall_fail() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal(Aggregation.ALL)
    repo.create(parent)
    repo.create(_child(parent.id, GoalStatus.DONE))
    repo.create(_child(parent.id, GoalStatus.FAILED))
    repo.create(_child(parent.id, GoalStatus.DONE))

    v = verify_composite(
        parent.predicate_of_success, _traj(parent.id), parent, goal_repo=repo
    )
    assert v.status == VerifyStatus.FAIL
    assert v.evidence[0]["failed"] == 1


def test_composite_all_with_abandoned_child_fail() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal(Aggregation.ALL)
    repo.create(parent)
    repo.create(_child(parent.id, GoalStatus.DONE))
    repo.create(_child(parent.id, GoalStatus.ABANDONED))

    v = verify_composite(
        parent.predicate_of_success, _traj(parent.id), parent, goal_repo=repo
    )
    assert v.status == VerifyStatus.FAIL


# ---------- aggregation=CUSTOM ----------


def _mock_judge(success: bool, reasoning: str = "ok") -> MagicMock:
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"success": '
                + ("true" if success else "false")
                + f', "reasoning": "{reasoning}"}}'
            )
        )
    ]
    response.usage = MagicMock(prompt_tokens=20, completion_tokens=30)
    return response


def test_composite_custom_judge_pass() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal(Aggregation.CUSTOM, custom_check="at least 2 of 3 succeed")
    repo.create(parent)
    repo.create(_child(parent.id, GoalStatus.DONE))
    repo.create(_child(parent.id, GoalStatus.DONE))
    repo.create(_child(parent.id, GoalStatus.FAILED))

    judge = MagicMock(return_value=_mock_judge(success=True, reasoning="2 of 3 ok"))
    v = verify_composite(
        parent.predicate_of_success,
        _traj(parent.id),
        parent,
        goal_repo=repo,
        llm_call=judge,
    )
    assert v.status == VerifyStatus.SUCCESS
    assert v.measured_by == "composite"
    # tier=light для судьи
    assert judge.call_args.kwargs["tier"] == "light"


def test_composite_custom_judge_fail() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal(Aggregation.CUSTOM, custom_check="majority")
    repo.create(parent)
    repo.create(_child(parent.id, GoalStatus.FAILED))
    repo.create(_child(parent.id, GoalStatus.FAILED))
    repo.create(_child(parent.id, GoalStatus.DONE))

    judge = MagicMock(return_value=_mock_judge(success=False, reasoning="too many failed"))
    v = verify_composite(
        parent.predicate_of_success,
        _traj(parent.id),
        parent,
        goal_repo=repo,
        llm_call=judge,
    )
    assert v.status == VerifyStatus.FAIL


def test_composite_custom_judge_unparseable_undetermined() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal(Aggregation.CUSTOM, custom_check="any")
    repo.create(parent)
    repo.create(_child(parent.id, GoalStatus.DONE))

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="garbage"))]
    response.usage = MagicMock(prompt_tokens=20, completion_tokens=30)

    v = verify_composite(
        parent.predicate_of_success,
        _traj(parent.id),
        parent,
        goal_repo=repo,
        llm_call=MagicMock(return_value=response),
    )
    assert v.status == VerifyStatus.UNDETERMINED


def test_composite_custom_judge_exception_undetermined() -> None:
    repo = GoalRepository(":memory:")
    parent = _parent_goal(Aggregation.CUSTOM)
    repo.create(parent)
    repo.create(_child(parent.id, GoalStatus.DONE))

    v = verify_composite(
        parent.predicate_of_success,
        _traj(parent.id),
        parent,
        goal_repo=repo,
        llm_call=MagicMock(side_effect=RuntimeError("net")),
    )
    assert v.status == VerifyStatus.UNDETERMINED


# ---------- Dispatcher integration ----------


def test_dispatcher_routes_to_composite_with_goal_repo() -> None:
    from harnes.metacycle.verifiers import verify

    repo = GoalRepository(":memory:")
    parent = _parent_goal(Aggregation.ALL)
    repo.create(parent)
    repo.create(_child(parent.id, GoalStatus.DONE))

    v = verify(_traj(parent.id), parent, goal_repo=repo)
    assert v.measured_by == "composite"
    assert v.status == VerifyStatus.SUCCESS


def test_dispatcher_composite_without_repo_undetermined() -> None:
    from harnes.metacycle.verifiers import verify

    parent = _parent_goal(Aggregation.ALL)
    v = verify(_traj(parent.id), parent)  # no goal_repo
    assert v.status == VerifyStatus.UNDETERMINED
