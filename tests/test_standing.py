"""Tests for harnes.metacycle.standing."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
)
from harnes.goals.store import GoalRepository
from harnes.metacycle.schema import FocusFrame, SalientItem
from harnes.metacycle.standing import (
    StandingContext,
    _has_active_child,
    bootstrap_starter_standing_goals,
    check_standing_goals,
    get_policy,
    list_policies,
    on_alert_observation,
    on_prev_verify_failure,
)


@pytest.fixture
def repo() -> GoalRepository:
    return GoalRepository(":memory:")


def _make_standing(policy_name: str, **metadata) -> Goal:
    return Goal(
        description=f"standing for {policy_name}",
        goal_class=GoalClass.STANDING,
        status=GoalStatus.ACTIVE,
        predicate_of_success=JudgePredicate(criterion="never"),
        origin=Origin.OPERATOR,
        originator="test",
        metadata={"policy_name": policy_name, **metadata},
    )


def _focus(urgencies: list[float]) -> FocusFrame:
    return FocusFrame(
        salient_items=[
            SalientItem(
                observation_id=Goal(  # any uuid
                    description="x",
                    goal_class=GoalClass.TASK,
                    predicate_of_success=JudgePredicate(criterion="x"),
                    origin=Origin.OPERATOR,
                    originator="test",
                ).id,
                relevance=1.0,
                novelty=0.5,
                urgency=u,
                score=u,
            )
            for u in urgencies
        ]
    )


# ---------- registry ----------


def test_registry_has_starter_policies() -> None:
    names = list_policies()
    assert "on_alert_observation" in names
    assert "on_prev_verify_failure" in names


def test_get_unknown_policy_returns_none() -> None:
    assert get_policy("nonsense") is None


# ---------- on_alert_observation ----------


def test_alert_policy_fires_on_high_urgency(repo: GoalRepository) -> None:
    parent = _make_standing("on_alert_observation", child_priority=4)
    repo.create(parent)
    ctx = StandingContext(tick_id=1, focus=_focus([0.95]), has_active_goal_now=False)

    child = on_alert_observation(ctx, parent, repo)
    assert child is not None
    assert child.goal_class == GoalClass.INQUIRY
    assert child.priority == 4
    assert child.origin == Origin.DECOMPOSITION
    assert child.parent_id == parent.id


def test_alert_policy_doesnt_fire_on_low_urgency(repo: GoalRepository) -> None:
    parent = _make_standing("on_alert_observation")
    repo.create(parent)
    ctx = StandingContext(tick_id=1, focus=_focus([0.3, 0.4]), has_active_goal_now=False)
    assert on_alert_observation(ctx, parent, repo) is None


def test_alert_policy_doesnt_fire_on_empty_focus(repo: GoalRepository) -> None:
    parent = _make_standing("on_alert_observation")
    repo.create(parent)
    ctx = StandingContext(tick_id=1, focus=FocusFrame(), has_active_goal_now=False)
    assert on_alert_observation(ctx, parent, repo) is None


# ---------- on_prev_verify_failure ----------


def test_failure_policy_fires_on_high_priority_failure(
    repo: GoalRepository,
) -> None:
    parent = _make_standing("on_prev_verify_failure", priority_threshold=2)
    repo.create(parent)

    failed = Goal(
        description="failed thing",
        goal_class=GoalClass.TASK,
        predicate_of_success=JudgePredicate(criterion="x"),
        priority=3,
        status=GoalStatus.FAILED,
        origin=Origin.OPERATOR,
        originator="test",
    )
    repo.create(failed)

    ctx = StandingContext(tick_id=1, focus=FocusFrame(), has_active_goal_now=False)
    child = on_prev_verify_failure(ctx, parent, repo)
    assert child is not None
    assert "failed thing" in child.description
    assert child.metadata["failed_goal_id"] == str(failed.id)


def test_failure_policy_skips_low_priority(repo: GoalRepository) -> None:
    parent = _make_standing("on_prev_verify_failure", priority_threshold=3)
    repo.create(parent)
    repo.create(
        Goal(
            description="minor",
            goal_class=GoalClass.TASK,
            predicate_of_success=JudgePredicate(criterion="x"),
            priority=1,
            status=GoalStatus.FAILED,
            origin=Origin.OPERATOR,
            originator="test",
        )
    )
    ctx = StandingContext(tick_id=1, focus=FocusFrame(), has_active_goal_now=False)
    assert on_prev_verify_failure(ctx, parent, repo) is None


def test_failure_policy_ignores_own_children(repo: GoalRepository) -> None:
    """Не должны порождать новую инкуайри из-за failed child своего же standing."""
    parent = _make_standing("on_prev_verify_failure", priority_threshold=1)
    repo.create(parent)

    failed_child = Goal(
        description="diagnosed and failed",
        goal_class=GoalClass.INQUIRY,
        predicate_of_success=JudgePredicate(criterion="x"),
        priority=2,
        status=GoalStatus.FAILED,
        origin=Origin.DECOMPOSITION,
        originator=f"standing:{parent.id}",
        parent_id=parent.id,
    )
    repo.create(failed_child)

    ctx = StandingContext(tick_id=1, focus=FocusFrame(), has_active_goal_now=False)
    assert on_prev_verify_failure(ctx, parent, repo) is None


# ---------- check_standing_goals dispatcher ----------


def test_check_iterates_active_standing(repo: GoalRepository) -> None:
    p1 = _make_standing("on_alert_observation")
    p2 = _make_standing("on_prev_verify_failure", priority_threshold=1)
    repo.create(p1)
    repo.create(p2)

    # Trigger alert path
    ctx = StandingContext(tick_id=1, focus=_focus([1.0]), has_active_goal_now=False)
    spawned = check_standing_goals(ctx, repo)
    assert len(spawned) == 1
    assert spawned[0].parent_id == p1.id


def test_check_dedups_when_active_child_exists(repo: GoalRepository) -> None:
    p1 = _make_standing("on_alert_observation")
    repo.create(p1)

    ctx = StandingContext(tick_id=1, focus=_focus([1.0]), has_active_goal_now=False)
    first = check_standing_goals(ctx, repo)
    assert len(first) == 1

    # Second tick — child уже pending/active, новый не создаётся
    second = check_standing_goals(ctx, repo)
    assert second == []


def test_check_ignores_unknown_policy(repo: GoalRepository) -> None:
    bad = _make_standing("nonsense_unregistered")
    repo.create(bad)
    ctx = StandingContext(tick_id=1, focus=_focus([1.0]), has_active_goal_now=False)
    spawned = check_standing_goals(ctx, repo)
    assert spawned == []


def test_check_skips_non_active_standing(repo: GoalRepository) -> None:
    g = _make_standing("on_alert_observation")
    g.status = GoalStatus.SUSPENDED
    repo.create(g)
    ctx = StandingContext(tick_id=1, focus=_focus([1.0]), has_active_goal_now=False)
    spawned = check_standing_goals(ctx, repo)
    assert spawned == []


# ---------- has_active_child ----------


def test_has_active_child_true_for_pending(repo: GoalRepository) -> None:
    parent = _make_standing("on_alert_observation")
    repo.create(parent)
    child = Goal(
        description="x",
        goal_class=GoalClass.INQUIRY,
        predicate_of_success=JudgePredicate(criterion="x"),
        status=GoalStatus.PENDING,
        origin=Origin.DECOMPOSITION,
        originator="test",
        parent_id=parent.id,
    )
    repo.create(child)
    assert _has_active_child(repo, parent.id) is True


def test_has_active_child_false_when_only_done(repo: GoalRepository) -> None:
    parent = _make_standing("on_alert_observation")
    repo.create(parent)
    child = Goal(
        description="x",
        goal_class=GoalClass.INQUIRY,
        predicate_of_success=JudgePredicate(criterion="x"),
        status=GoalStatus.DONE,
        origin=Origin.DECOMPOSITION,
        originator="test",
        parent_id=parent.id,
    )
    repo.create(child)
    assert _has_active_child(repo, parent.id) is False


# ---------- bootstrap ----------


def test_bootstrap_creates_starter_set(repo: GoalRepository) -> None:
    created = bootstrap_starter_standing_goals(repo)
    assert len(created) == 2
    all_standing = repo.list_by_class(GoalClass.STANDING)
    assert len(all_standing) == 2
    policies = {g.metadata["policy_name"] for g in all_standing}
    assert policies == {"on_alert_observation", "on_prev_verify_failure"}


def test_bootstrap_idempotent(repo: GoalRepository) -> None:
    bootstrap_starter_standing_goals(repo)
    second = bootstrap_starter_standing_goals(repo)
    assert second == []
    # Всё ещё 2, не 4
    assert len(repo.list_by_class(GoalClass.STANDING)) == 2
