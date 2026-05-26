"""Tests for harnes.goals.store.

Каждый тест использует свою in-memory SQLite БД через fixture.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from harnes.goals.schema import (
    Aggregation,
    Budget,
    CompositePredicate,
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
    StateChangePredicate,
    StructuralPredicate,
)
from harnes.goals.store import GoalRepository


@pytest.fixture
def repo() -> GoalRepository:
    """In-memory репозиторий — изолированный per-test."""
    return GoalRepository(":memory:")


def _make_goal(
    description: str = "test goal",
    goal_class: GoalClass = GoalClass.TASK,
    status: GoalStatus = GoalStatus.PENDING,
    origin: Origin = Origin.OPERATOR,
    parent_id=None,
) -> Goal:
    return Goal(
        description=description,
        goal_class=goal_class,
        predicate_of_success=JudgePredicate(criterion="ok if X"),
        status=status,
        origin=origin,
        originator="test",
        parent_id=parent_id,
    )


def test_create_and_get_roundtrip(repo: GoalRepository) -> None:
    goal = _make_goal("write a file")
    repo.create(goal)

    fetched = repo.get(goal.id)
    assert fetched is not None
    assert fetched.id == goal.id
    assert fetched.description == "write a file"
    assert fetched.goal_class == GoalClass.TASK


def test_get_missing_returns_none(repo: GoalRepository) -> None:
    assert repo.get(uuid4()) is None


def test_predicate_preserved_through_storage(repo: GoalRepository) -> None:
    """Discriminated union должен пережить SQLite roundtrip."""
    goal = Goal(
        description="ensure file exists",
        goal_class=GoalClass.TASK,
        predicate_of_success=StateChangePredicate(
            check_tool_id="file_exists",
            expected_outcome={"exists": True},
        ),
        origin=Origin.OPERATOR,
        originator="test",
    )
    repo.create(goal)
    fetched = repo.get(goal.id)
    assert fetched is not None
    assert isinstance(fetched.predicate_of_success, StateChangePredicate)
    assert fetched.predicate_of_success.check_tool_id == "file_exists"


def test_composite_predicate_preserved(repo: GoalRepository) -> None:
    goal = Goal(
        description="parent",
        goal_class=GoalClass.TASK,
        predicate_of_success=CompositePredicate(aggregation=Aggregation.ALL),
        origin=Origin.OPERATOR,
        originator="test",
    )
    repo.create(goal)
    fetched = repo.get(goal.id)
    assert fetched is not None
    assert isinstance(fetched.predicate_of_success, CompositePredicate)
    assert fetched.predicate_of_success.aggregation == Aggregation.ALL


def test_budget_preserved(repo: GoalRepository) -> None:
    goal = _make_goal()
    goal.budget = Budget(tokens=1000, tokens_consumed=250)
    repo.create(goal)
    fetched = repo.get(goal.id)
    assert fetched is not None
    assert fetched.budget.tokens == 1000
    assert fetched.budget.tokens_consumed == 250


def test_depends_on_preserved(repo: GoalRepository) -> None:
    dep_id = uuid4()
    goal = _make_goal()
    goal.depends_on = [dep_id]
    repo.create(goal)
    fetched = repo.get(goal.id)
    assert fetched is not None
    assert fetched.depends_on == [dep_id]


def test_metadata_preserved(repo: GoalRepository) -> None:
    goal = _make_goal()
    goal.metadata = {"source": "operator", "tags": ["important"]}
    repo.create(goal)
    fetched = repo.get(goal.id)
    assert fetched is not None
    assert fetched.metadata["source"] == "operator"
    assert fetched.metadata["tags"] == ["important"]


def test_update_changes_status(repo: GoalRepository) -> None:
    goal = _make_goal()
    repo.create(goal)

    goal.status = GoalStatus.ACTIVE
    repo.update(goal)

    fetched = repo.get(goal.id)
    assert fetched is not None
    assert fetched.status == GoalStatus.ACTIVE


def test_update_missing_raises(repo: GoalRepository) -> None:
    goal = _make_goal()
    with pytest.raises(KeyError):
        repo.update(goal)  # not created yet


def test_list_by_status(repo: GoalRepository) -> None:
    repo.create(_make_goal("a", status=GoalStatus.PENDING))
    repo.create(_make_goal("b", status=GoalStatus.ACTIVE))
    repo.create(_make_goal("c", status=GoalStatus.PENDING))

    pending = repo.list_by_status(GoalStatus.PENDING)
    active = repo.list_by_status(GoalStatus.ACTIVE)
    assert len(pending) == 2
    assert len(active) == 1


def test_list_children(repo: GoalRepository) -> None:
    parent = _make_goal("parent")
    repo.create(parent)

    child1 = _make_goal("child1", parent_id=parent.id)
    child2 = _make_goal("child2", parent_id=parent.id)
    other = _make_goal("other")
    repo.create(child1)
    repo.create(child2)
    repo.create(other)

    children = repo.list_children(parent.id)
    assert {c.description for c in children} == {"child1", "child2"}


# ---------- Operator-approval flow ----------


def test_approve_flow(repo: GoalRepository) -> None:
    goal = _make_goal(status=GoalStatus.PENDING_APPROVAL)
    repo.create(goal)

    approved = repo.approve(goal.id)
    assert approved.status == GoalStatus.PENDING


def test_reject_flow_sets_reason(repo: GoalRepository) -> None:
    goal = _make_goal(status=GoalStatus.PENDING_APPROVAL)
    repo.create(goal)

    rejected = repo.reject(goal.id, "out of scope")
    assert rejected.status == GoalStatus.ABANDONED
    assert rejected.metadata["reject_reason"] == "out of scope"


def test_approve_wrong_status_raises(repo: GoalRepository) -> None:
    goal = _make_goal(status=GoalStatus.ACTIVE)
    repo.create(goal)
    with pytest.raises(ValueError):
        repo.approve(goal.id)


def test_list_pending_approval(repo: GoalRepository) -> None:
    repo.create(_make_goal("p1", status=GoalStatus.PENDING_APPROVAL))
    repo.create(_make_goal("p2", status=GoalStatus.PENDING_APPROVAL))
    repo.create(_make_goal("active", status=GoalStatus.ACTIVE))

    pending = repo.list_pending_approval()
    assert len(pending) == 2


# ---------- Pending verifications ----------


def test_pending_verification_register_and_list(repo: GoalRepository) -> None:
    goal = _make_goal()
    repo.create(goal)

    repo.register_pending_verification(goal.id, "webhook from CI")
    pending = repo.list_pending_verifications()
    assert len(pending) == 1
    assert pending[0].goal_id == goal.id
    assert pending[0].expected_signal == "webhook from CI"
    assert pending[0].resolved_at is None


def test_pending_verification_resolve(repo: GoalRepository) -> None:
    goal = _make_goal()
    repo.create(goal)
    repo.register_pending_verification(goal.id, "signal")

    pending = repo.list_pending_verifications()
    pv_id = pending[0].id
    assert pv_id is not None

    repo.resolve_verification(pv_id, "success")
    assert repo.list_pending_verifications() == []
