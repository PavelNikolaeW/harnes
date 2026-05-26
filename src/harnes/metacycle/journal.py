"""Tick journal — persistent log событий метацикла + периодические snapshot'ы.

См. v1.0 #35.

Зачем:
- run-loop --real должен пройти сутки+ без потери состояния.
- При крэше нужен recovery: с какого tick_id продолжить, какие счётчики.
- Для длинного horizon-замера нужна observability — "что агент делал в Х часу".

Архитектура: один SQLite файл с двумя таблицами:
- tick_events    — append-only log событий (tick_started, goal_spawned, ...)
- tick_snapshots — снимки счётчиков каждые N тиков; для resume.

Falls open: при ошибках записи событие логируется в structlog, но run-loop
продолжает работать (не блокируем main цикл).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import structlog
from sqlmodel import Field, Session, SQLModel, create_engine, select

log = structlog.get_logger()


# ---------- Event types ----------


class TickEventType(str, Enum):
    """Категории событий, которые run-loop пишет в journal.

    Минимальный набор для observability — расширяется по мере надобности.
    """

    TICK_STARTED = "tick_started"
    TICK_DONE = "tick_done"
    TICK_IDLE = "tick_idle"
    GOAL_ACTIVATED = "goal_activated"
    GOAL_COMPLETED = "goal_completed"
    GOAL_FAILED = "goal_failed"
    GOAL_SPAWNED = "goal_spawned"  # self-generated (standing / reflect inquiry)
    SKILL_VERSIONED = "skill_versioned"  # reflect → new skill version
    ERROR = "error"
    LOOP_STARTED = "loop_started"
    LOOP_STOPPED = "loop_stopped"


# ---------- Rows ----------


class TickEventRow(SQLModel, table=True):
    """Одна запись append-only journal'а."""

    __tablename__ = "tick_events"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime
    tick_id: int = Field(index=True)
    event_type: str = Field(index=True)
    payload_json: str = "{}"


class TickSnapshotRow(SQLModel, table=True):
    """Снимок состояния счётчиков для recovery."""

    __tablename__ = "tick_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime
    tick_id: int = Field(index=True)

    processed_count: int = 0
    idle_count: int = 0
    error_count: int = 0
    ticks_with_self_spawn: int = 0
    total_self_spawned: int = 0
    skill_versions_count: int = 0  # сколько раз reflect bump'нул скилл
    extra_json: str = "{}"  # для будущих счётчиков


# ---------- Store ----------


class TickJournal:
    """SQLite-репозиторий tick-журнала."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        url = self._build_url(db_path)
        self.engine = create_engine(url, echo=False)
        SQLModel.metadata.create_all(self.engine)

    @staticmethod
    def _build_url(db_path: Path | str) -> str:
        if db_path == ":memory:":
            return "sqlite:///:memory:"
        p = Path(db_path).resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Read-only filesystem / permissions / etc — falls open в :memory:.
            log.warning(
                "journal.path.unavailable",
                path=str(p),
                error=str(exc),
                fallback="in-memory",
            )
            return "sqlite:///:memory:"
        return f"sqlite:///{p}"

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self.engine) as s:
            yield s

    # ---------- Append ----------

    def append(
        self,
        tick_id: int,
        event_type: TickEventType,
        payload: dict[str, Any] | None = None,
    ) -> TickEventRow | None:
        """Append-only event. Возвращает None при ошибке (falls open)."""
        try:
            row = TickEventRow(
                timestamp=datetime.now(UTC),
                tick_id=tick_id,
                event_type=event_type.value,
                payload_json=json.dumps(payload or {}, default=str),
            )
            with self._session() as s:
                s.add(row)
                s.commit()
                s.refresh(row)
            return row
        except Exception as exc:  # noqa: BLE001 — journal не должен валить loop
            log.warning(
                "journal.append.failed",
                event_type=event_type.value,
                tick_id=tick_id,
                error=str(exc),
            )
            return None

    # ---------- Snapshot ----------

    def snapshot(
        self,
        tick_id: int,
        processed_count: int,
        idle_count: int,
        error_count: int = 0,
        ticks_with_self_spawn: int = 0,
        total_self_spawned: int = 0,
        skill_versions_count: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> TickSnapshotRow | None:
        try:
            row = TickSnapshotRow(
                timestamp=datetime.now(UTC),
                tick_id=tick_id,
                processed_count=processed_count,
                idle_count=idle_count,
                error_count=error_count,
                ticks_with_self_spawn=ticks_with_self_spawn,
                total_self_spawned=total_self_spawned,
                skill_versions_count=skill_versions_count,
                extra_json=json.dumps(extra or {}, default=str),
            )
            with self._session() as s:
                s.add(row)
                s.commit()
                s.refresh(row)
            return row
        except Exception as exc:  # noqa: BLE001
            log.warning("journal.snapshot.failed", tick_id=tick_id, error=str(exc))
            return None

    # ---------- Queries ----------

    def latest_snapshot(self) -> TickSnapshotRow | None:
        """Последний snapshot — точка восстановления."""
        with self._session() as s:
            q = select(TickSnapshotRow).order_by(TickSnapshotRow.id.desc()).limit(1)
            rows = list(s.exec(q).all())
            return rows[0] if rows else None

    def recent_events(
        self,
        limit: int = 50,
        event_type: TickEventType | None = None,
        tick_id: int | None = None,
        since: datetime | None = None,
    ) -> list[TickEventRow]:
        with self._session() as s:
            q = select(TickEventRow).order_by(TickEventRow.id.desc())
            if event_type is not None:
                q = q.where(TickEventRow.event_type == event_type.value)
            if tick_id is not None:
                q = q.where(TickEventRow.tick_id == tick_id)
            if since is not None:
                q = q.where(TickEventRow.timestamp >= since)
            q = q.limit(limit)
            return list(s.exec(q).all())

    def event_count(self, event_type: TickEventType | None = None) -> int:
        with self._session() as s:
            from sqlalchemy import func

            q = select(func.count(TickEventRow.id))
            if event_type is not None:
                q = q.where(TickEventRow.event_type == event_type.value)
            return int(s.exec(q).one())

    def stats(self) -> dict[str, Any]:
        """Сводная статистика по journal'у."""
        with self._session() as s:
            from sqlalchemy import func

            total_events = int(s.exec(select(func.count(TickEventRow.id))).one())
            total_snapshots = int(s.exec(select(func.count(TickSnapshotRow.id))).one())

            # Counts per event_type
            type_rows = list(
                s.exec(
                    select(TickEventRow.event_type, func.count(TickEventRow.id))
                    .group_by(TickEventRow.event_type)
                ).all()
            )
            by_type: dict[str, int] = {}
            for entry in type_rows:
                if isinstance(entry, tuple):
                    t, c = entry
                else:
                    # SQLAlchemy 2.x: возвращает Row → t/c индексами.
                    t, c = entry[0], entry[1]
                by_type[str(t)] = int(c)

            # Min/max tick_id
            min_tick = s.exec(select(func.min(TickEventRow.tick_id))).first()
            max_tick = s.exec(select(func.max(TickEventRow.tick_id))).first()

        return {
            "total_events": total_events,
            "total_snapshots": total_snapshots,
            "by_event_type": by_type,
            "min_tick_id": min_tick,
            "max_tick_id": max_tick,
        }
