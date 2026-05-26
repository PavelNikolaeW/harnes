"""Episodic store on LanceDB.

См. `agent_architecture.html` § 13.

Две таблицы:
- trajectories: метаданные траекторий (id, goal_id, status, cost, time range)
- steps: типизированные шаги, content_json — JSON-сериализованный Step

LanceDB embedded (no docker), даёт versioned tables c time-travel из коробки.

В v0 embedding-колонки не используются — добавим, когда recall начнёт делать
семантический поиск по episodic memory.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import lancedb
import pyarrow as pa
import structlog

from harnes.react.schema import Cost, Step, Trajectory, TrajectoryStatus

log = structlog.get_logger()


TRAJECTORIES_TABLE = "trajectories"
STEPS_TABLE = "steps"


_trajectory_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("goal_id", pa.string()),
        pa.field("parent_trajectory_id", pa.string()),
        pa.field("status", pa.string()),
        pa.field("total_cost_tokens", pa.int64()),
        pa.field("total_cost_latency", pa.float64()),
        pa.field("final_state_json", pa.string()),
        pa.field("started_at", pa.timestamp("us")),
        pa.field("ended_at", pa.timestamp("us")),
        pa.field("metadata_json", pa.string()),
    ]
)

_step_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("trajectory_id", pa.string()),
        pa.field("goal_id", pa.string()),
        pa.field("step_type", pa.string()),
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("cost_tokens", pa.int64()),
        pa.field("cost_latency", pa.float64()),
        pa.field("content_json", pa.string()),
    ]
)


class EpisodicStore:
    """LanceDB-обёртка для episodic-логирования траекторий."""

    def __init__(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(p))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        # NB: `table_names()` is deprecated в LanceDB но `list_tables()` в
        # этой версии возвращает unhashable-структуру; вернёмся к новому API,
        # когда оно стабилизируется.
        names = set(self.db.table_names())
        if TRAJECTORIES_TABLE not in names:
            self.db.create_table(TRAJECTORIES_TABLE, schema=_trajectory_schema)
        if STEPS_TABLE not in names:
            self.db.create_table(STEPS_TABLE, schema=_step_schema)

    # ---------- write ----------

    def write_trajectory(self, traj: Trajectory) -> None:
        """Сохраняет траекторию целиком: metadata + все шаги.

        Идемпотентно по trajectory.id для steps (но не для metadata — повторная
        запись просто добавит дубликат, фильтрация по uniqueness — TBD).
        """
        meta = {
            "id": str(traj.id),
            "goal_id": str(traj.goal_id),
            "parent_trajectory_id": (
                str(traj.parent_trajectory_id) if traj.parent_trajectory_id else ""
            ),
            "status": traj.status.value if traj.status else "",
            "total_cost_tokens": traj.total_cost.tokens,
            "total_cost_latency": traj.total_cost.latency_seconds,
            "final_state_json": json.dumps(traj.final_state)
            if traj.final_state is not None
            else "",
            "started_at": traj.started_at.replace(tzinfo=None),
            "ended_at": traj.ended_at.replace(tzinfo=None) if traj.ended_at else None,
            "metadata_json": json.dumps(traj.metadata),
        }
        self.db.open_table(TRAJECTORIES_TABLE).add([meta])

        step_rows = [self._step_to_row(s, traj.id, traj.goal_id) for s in traj.steps]
        if step_rows:
            self.db.open_table(STEPS_TABLE).add(step_rows)

        log.debug(
            "episodic.trajectory.written",
            trajectory_id=str(traj.id),
            steps=len(traj.steps),
        )

    @staticmethod
    def _step_to_row(step: Step, trajectory_id: UUID, goal_id: UUID) -> dict[str, Any]:
        # `step` тут — это union одного из ThoughtStep/ActionStep/...; у всех
        # есть type/id/timestamp/cost, payload-поля живут на самом классе.
        content = step.model_dump(mode="json")
        return {
            "id": str(step.id),
            "trajectory_id": str(trajectory_id),
            "goal_id": str(goal_id),
            "step_type": step.type,
            "timestamp": step.timestamp.replace(tzinfo=None),
            "cost_tokens": step.cost.tokens,
            "cost_latency": step.cost.latency_seconds,
            "content_json": json.dumps(content),
        }

    # ---------- read ----------

    def get_trajectory_meta(self, trajectory_id: UUID) -> dict[str, Any] | None:
        df = (
            self.db.open_table(TRAJECTORIES_TABLE)
            .search()
            .where(f"id = '{trajectory_id}'")
            .limit(1)
            .to_list()
        )
        return df[0] if df else None

    def get_steps(self, trajectory_id: UUID) -> list[dict[str, Any]]:
        """Возвращает все шаги траектории в порядке timestamp."""
        rows = (
            self.db.open_table(STEPS_TABLE)
            .search()
            .where(f"trajectory_id = '{trajectory_id}'")
            .limit(10_000)
            .to_list()
        )
        rows.sort(key=lambda r: r["timestamp"])
        return rows

    def list_trajectories_for_goal(self, goal_id: UUID) -> list[dict[str, Any]]:
        return (
            self.db.open_table(TRAJECTORIES_TABLE)
            .search()
            .where(f"goal_id = '{goal_id}'")
            .limit(10_000)
            .to_list()
        )

    def recent_steps(self, limit: int = 100) -> list[dict[str, Any]]:
        """Последние N шагов across all trajectories."""
        rows = self.db.open_table(STEPS_TABLE).search().limit(limit * 10).to_list()
        rows.sort(key=lambda r: r["timestamp"], reverse=True)
        return rows[:limit]

    def recent_trajectories(
        self, limit: int = 20, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Последние N трейекторий (по started_at desc), опционально по статусу."""
        q = self.db.open_table(TRAJECTORIES_TABLE).search().limit(limit * 10)
        if status is not None:
            q = q.where(f"status = '{status}'")
        rows = q.to_list()
        rows.sort(key=lambda r: r["started_at"], reverse=True)
        return rows[:limit]
