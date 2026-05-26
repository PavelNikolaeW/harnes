"""Goal store on SQLite via SQLModel.

См. `agent_architecture.html` § 4.

Goal-объекты сериализуются в табличку `goals`. Разнотипные/полиморфные поля
(predicate, budget, depends_on, metadata) хранятся как JSON-строки в TEXT-колонках.

Pending verifications живут в отдельной таблице `pending_verifications`.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator
from uuid import UUID

from pydantic import TypeAdapter
from sqlmodel import Field, Session, SQLModel, create_engine, select

from harnes.goals.schema import (
    Aggregation,
    Budget,
    Goal,
    GoalClass,
    GoalStatus,
    Origin,
    OriginSubtype,
    PredicateOfSuccess,
)


_predicate_adapter: TypeAdapter[PredicateOfSuccess] = TypeAdapter(PredicateOfSuccess)


# ---------- Row models ----------


class GoalRow(SQLModel, table=True):
    """Табличное представление Goal."""

    __tablename__ = "goals"

    id: UUID = Field(primary_key=True)
    parent_id: UUID | None = Field(default=None, foreign_key="goals.id")

    description: str
    goal_class: GoalClass
    status: GoalStatus
    priority: int = 0
    deadline: datetime | None = None

    origin: Origin
    originator: str
    origin_subtype: OriginSubtype | None = None
    aggregation: Aggregation | None = None

    created_at: datetime
    updated_at: datetime

    # JSON-сериализованные поля
    predicate_json: str
    budget_json: str
    allowed_skills_json: str = "[]"
    depends_on_json: str = "[]"
    metadata_json: str = "{}"


class PendingVerificationRow(SQLModel, table=True):
    """Отложенная верификация для composite/external предикатов."""

    __tablename__ = "pending_verifications"

    id: int | None = Field(default=None, primary_key=True)
    goal_id: UUID = Field(foreign_key="goals.id")
    expected_signal: str
    registered_at: datetime
    resolved_at: datetime | None = None
    resolved_status: str | None = None


# ---------- Repository ----------


class GoalRepository:
    """CRUD + queries для Goal-объектов поверх SQLite."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        url = self._build_url(db_path)
        self.engine = create_engine(url, echo=False)
        SQLModel.metadata.create_all(self.engine)

    @staticmethod
    def _build_url(db_path: Path | str) -> str:
        if db_path == ":memory:":
            return "sqlite:///:memory:"
        p = Path(db_path).resolve()
        return f"sqlite:///{p}"

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self.engine) as s:
            yield s

    # ---------- Mapping Goal ↔ GoalRow ----------

    @staticmethod
    def _to_row(goal: Goal) -> GoalRow:
        return GoalRow(
            id=goal.id,
            parent_id=goal.parent_id,
            description=goal.description,
            goal_class=goal.goal_class,
            status=goal.status,
            priority=goal.priority,
            deadline=goal.deadline,
            origin=goal.origin,
            originator=goal.originator,
            origin_subtype=goal.origin_subtype,
            aggregation=goal.aggregation,
            created_at=goal.created_at,
            updated_at=goal.updated_at,
            predicate_json=goal.predicate_of_success.model_dump_json(),
            budget_json=goal.budget.model_dump_json(),
            allowed_skills_json=json.dumps(goal.allowed_skills),
            depends_on_json=json.dumps([str(u) for u in goal.depends_on]),
            metadata_json=json.dumps(goal.metadata),
        )

    @staticmethod
    def _from_row(row: GoalRow) -> Goal:
        return Goal(
            id=row.id,
            parent_id=row.parent_id,
            description=row.description,
            goal_class=row.goal_class,
            status=row.status,
            priority=row.priority,
            deadline=row.deadline,
            origin=row.origin,
            originator=row.originator,
            origin_subtype=row.origin_subtype,
            aggregation=row.aggregation,
            created_at=row.created_at,
            updated_at=row.updated_at,
            predicate_of_success=_predicate_adapter.validate_json(row.predicate_json),
            budget=Budget.model_validate_json(row.budget_json),
            allowed_skills=json.loads(row.allowed_skills_json),
            depends_on=[UUID(s) for s in json.loads(row.depends_on_json)],
            metadata=json.loads(row.metadata_json),
        )

    # ---------- CRUD ----------

    def create(self, goal: Goal) -> Goal:
        with self._session() as s:
            s.add(self._to_row(goal))
            s.commit()
        return goal

    def get(self, goal_id: UUID) -> Goal | None:
        with self._session() as s:
            row = s.get(GoalRow, goal_id)
            return self._from_row(row) if row else None

    def update(self, goal: Goal) -> Goal:
        goal.updated_at = datetime.now(UTC)
        new_row = self._to_row(goal)
        with self._session() as s:
            existing = s.get(GoalRow, goal.id)
            if existing is None:
                raise KeyError(f"Goal {goal.id} not found")
            # Перенос всех non-PK полей.
            for field_name in (
                "parent_id",
                "description",
                "goal_class",
                "status",
                "priority",
                "deadline",
                "origin",
                "originator",
                "origin_subtype",
                "aggregation",
                "updated_at",
                "predicate_json",
                "budget_json",
                "allowed_skills_json",
                "depends_on_json",
                "metadata_json",
            ):
                setattr(existing, field_name, getattr(new_row, field_name))
            s.add(existing)
            s.commit()
        return goal

    # ---------- Queries ----------

    def list_by_status(self, status: GoalStatus) -> list[Goal]:
        with self._session() as s:
            rows = s.exec(select(GoalRow).where(GoalRow.status == status)).all()
            return [self._from_row(r) for r in rows]

    def list_children(self, parent_id: UUID) -> list[Goal]:
        with self._session() as s:
            rows = s.exec(
                select(GoalRow).where(GoalRow.parent_id == parent_id)
            ).all()
            return [self._from_row(r) for r in rows]

    def list_pending_approval(self) -> list[Goal]:
        return self.list_by_status(GoalStatus.PENDING_APPROVAL)

    # ---------- Operator-approval flow ----------

    def approve(self, goal_id: UUID) -> Goal:
        goal = self.get(goal_id)
        if goal is None:
            raise KeyError(f"Goal {goal_id} not found")
        if goal.status != GoalStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Goal {goal_id} status={goal.status}, expected pending_approval"
            )
        goal.status = GoalStatus.PENDING
        return self.update(goal)

    def reject(self, goal_id: UUID, reason: str) -> Goal:
        goal = self.get(goal_id)
        if goal is None:
            raise KeyError(f"Goal {goal_id} not found")
        if goal.status != GoalStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Goal {goal_id} status={goal.status}, expected pending_approval"
            )
        goal.status = GoalStatus.ABANDONED
        goal.metadata = {**goal.metadata, "reject_reason": reason}
        return self.update(goal)

    # ---------- Pending verifications ----------

    def register_pending_verification(
        self, goal_id: UUID, expected_signal: str
    ) -> None:
        with self._session() as s:
            s.add(
                PendingVerificationRow(
                    goal_id=goal_id,
                    expected_signal=expected_signal,
                    registered_at=datetime.now(UTC),
                )
            )
            s.commit()

    def list_pending_verifications(self) -> list[PendingVerificationRow]:
        with self._session() as s:
            rows = s.exec(
                select(PendingVerificationRow).where(
                    PendingVerificationRow.resolved_at.is_(None)
                )
            ).all()
            return list(rows)

    def resolve_verification(self, pv_id: int, status: str) -> None:
        with self._session() as s:
            row = s.get(PendingVerificationRow, pv_id)
            if row is None:
                raise KeyError(f"PendingVerification {pv_id} not found")
            row.resolved_at = datetime.now(UTC)
            row.resolved_status = status
            s.add(row)
            s.commit()
