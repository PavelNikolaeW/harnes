"""Behaviour tests для /goals — create/approve/reject/abandon/bulk/budget.

Каждый тест проверяет: HTTP-статус и что state в БД изменился ожидаемым
образом (читаем напрямую через `goal_repo` fixture).
"""
from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient

from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
)
from harnes.goals.store import GoalRepository


def test_create_goal_via_form(client: TestClient, goal_repo: GoalRepository) -> None:
    """POST /goals form → 303 → /goals/{id}; repo.get(id) возвращает goal."""
    r = client.post(
        "/goals",
        data={
            "description": "new goal via form",
            "goal_class": "task",
            "priority": "0",
            "predicate_kind": "judge",
            "criterion": "done if everyone is happy",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("/goals/")
    goal_id = UUID(loc.rsplit("/", 1)[-1])
    goal = goal_repo.get(goal_id)
    assert goal is not None
    assert goal.description == "new goal via form"
    assert goal.origin == Origin.OPERATOR


def test_approve_pending_approval(
    client: TestClient,
    goal_repo: GoalRepository,
    pending_approval_goal: Goal,
) -> None:
    """POST /goals/{id}/approve → 303; status PENDING_APPROVAL → PENDING."""
    r = client.post(
        f"/goals/{pending_approval_goal.id}/approve", follow_redirects=False
    )
    assert r.status_code == 303
    fresh = goal_repo.get(pending_approval_goal.id)
    assert fresh is not None
    assert fresh.status == GoalStatus.PENDING


def test_reject_pending_approval(
    client: TestClient,
    goal_repo: GoalRepository,
    pending_approval_goal: Goal,
) -> None:
    """POST /goals/{id}/reject → 303; status → ABANDONED; metadata.reject_reason."""
    r = client.post(
        f"/goals/{pending_approval_goal.id}/reject",
        data={"reason": "not aligned"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    fresh = goal_repo.get(pending_approval_goal.id)
    assert fresh is not None
    assert fresh.status == GoalStatus.ABANDONED
    assert fresh.metadata.get("reject_reason") == "not aligned"


def test_abandon_active(
    client: TestClient,
    goal_repo: GoalRepository,
    pending_goal: Goal,
) -> None:
    """POST /goals/{id}/abandon non-terminal → 303; ABANDONED + reason."""
    r = client.post(
        f"/goals/{pending_goal.id}/abandon",
        data={"reason": "obsolete"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    fresh = goal_repo.get(pending_goal.id)
    assert fresh is not None
    assert fresh.status == GoalStatus.ABANDONED
    assert fresh.metadata.get("abandon_reason") == "obsolete"


def test_abandon_terminal_400(
    client: TestClient, goal_repo: GoalRepository
) -> None:
    """abandon DONE goal → 400 (terminal status frozen)."""
    g = Goal(
        description="already done",
        goal_class=GoalClass.TASK,
        predicate_of_success=JudgePredicate(criterion="x"),
        status=GoalStatus.DONE,
        origin=Origin.OPERATOR,
        originator="test",
    )
    goal_repo.create(g)
    r = client.post(f"/goals/{g.id}/abandon", follow_redirects=False)
    assert r.status_code == 400


def test_bulk_action_approve(
    client: TestClient, goal_repo: GoalRepository
) -> None:
    """3 PENDING_APPROVAL → bulk approve → все 3 в PENDING."""
    ids: list[str] = []
    for i in range(3):
        g = Goal(
            description=f"bulk goal {i}",
            goal_class=GoalClass.TASK,
            predicate_of_success=JudgePredicate(criterion="x"),
            status=GoalStatus.PENDING_APPROVAL,
            origin=Origin.OPERATOR,
            originator="test",
        )
        goal_repo.create(g)
        ids.append(str(g.id))

    r = client.post(
        "/goals/bulk",
        data={"action": "approve", "ids": ",".join(ids)},
        follow_redirects=False,
    )
    assert r.status_code == 303
    for sid in ids:
        fresh = goal_repo.get(UUID(sid))
        assert fresh is not None
        assert fresh.status == GoalStatus.PENDING


def test_budget_edit(
    client: TestClient,
    goal_repo: GoalRepository,
    pending_goal: Goal,
) -> None:
    """POST /goals/{id}/budget tokens=1000 → 303; budget.tokens=1000, 1 edit entry."""
    r = client.post(
        f"/goals/{pending_goal.id}/budget",
        data={"tokens": "1000", "reason": "more headroom"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    fresh = goal_repo.get(pending_goal.id)
    assert fresh is not None
    assert fresh.budget.tokens == 1000
    edits = fresh.metadata.get("budget_edits", [])
    assert len(edits) == 1
    assert edits[0]["new_limits"]["tokens"] == 1000


def test_budget_edit_negative_400(
    client: TestClient,
    goal_repo: GoalRepository,
    pending_goal: Goal,
) -> None:
    """tokens=-1 → 400 (rejected by _clean_limit)."""
    r = client.post(
        f"/goals/{pending_goal.id}/budget",
        data={"tokens": "-1"},
        follow_redirects=False,
    )
    assert r.status_code == 400
