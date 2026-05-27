"""Web→agent IPC: append-only лог команд от admin-консоли.

Зачем: webui позволяет оператору pause/resume run-loop и triger one-off тик,
не убивая контейнер. Этим занимается отдельная таблица в своём SQLite —
чтобы не мешать goal'ам и журналу. Run-loop в начале каждой итерации
drain'ит unconsumed команды и применяет их перед `sense`.

Команды v1:
- `pause`         — run-loop не делает тики, только sleep + drain.
- `resume`        — снять pause.
- `trigger_tick`  — выполнить один тик прямо сейчас (даже на паузе).

Дизайн append-only: команда сохраняется навсегда (audit trail), `consumed_at`
помечает обработку. Это даёт history для webui + reproducibility прогона.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from sqlmodel import Field, Session, SQLModel, create_engine, select

log = structlog.get_logger()


class CommandType(str, Enum):
    """Каноничные имена команд. UI и run-loop ссылаются на эти константы."""

    PAUSE = "pause"
    RESUME = "resume"
    TRIGGER_TICK = "trigger_tick"


class ConsumedStatus(str, Enum):
    OK = "ok"
    IGNORED = "ignored"
    ERROR = "error"


class WebCommandRow(SQLModel, table=True):
    """Одна команда оператора. Append-only."""

    __tablename__ = "web_commands"

    id: int | None = Field(default=None, primary_key=True)
    issued_at: datetime
    command: str = Field(index=True)
    issuer: str = "webui"
    payload_json: str = "{}"

    consumed_at: datetime | None = None
    consumed_status: str | None = None
    result_json: str = "{}"


class CommandStore:
    """SQLite-репозиторий web-команд. Безопасен для concurrent issue/drain."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.engine = create_engine(self._build_url(db_path), echo=False)
        SQLModel.metadata.create_all(self.engine)

    @staticmethod
    def _build_url(db_path: Path | str) -> str:
        if db_path == ":memory:":
            return "sqlite:///:memory:"
        p = Path(db_path).resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("commands.path.unavailable", path=str(p), error=str(exc))
            return "sqlite:///:memory:"
        return f"sqlite:///{p}"

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self.engine) as s:
            yield s

    # ---------- write ----------

    def issue(
        self,
        command: str | CommandType,
        payload: dict[str, Any] | None = None,
        issuer: str = "webui",
    ) -> WebCommandRow:
        """Выставить команду. Возвращает persisted row."""
        cmd_value = command.value if isinstance(command, CommandType) else str(command)
        row = WebCommandRow(
            issued_at=datetime.now(UTC),
            command=cmd_value,
            issuer=issuer,
            payload_json=json.dumps(payload or {}, default=str),
        )
        with self._session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
        log.info("commands.issued", id=row.id, command=cmd_value, issuer=issuer)
        return row

    def mark_consumed(
        self,
        command_id: int,
        status: ConsumedStatus | str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Пометить команду обработанной (idempotent — если уже consumed, no-op)."""
        st = status.value if isinstance(status, ConsumedStatus) else str(status)
        with self._session() as s:
            row = s.get(WebCommandRow, command_id)
            if row is None or row.consumed_at is not None:
                return
            row.consumed_at = datetime.now(UTC)
            row.consumed_status = st
            row.result_json = json.dumps(result or {}, default=str)
            s.add(row)
            s.commit()

    # ---------- read ----------

    def drain(self, limit: int = 50) -> list[WebCommandRow]:
        """Список unconsumed-команд по FIFO (issued_at asc). НЕ помечает consumed —
        вызывающий обязан вызвать mark_consumed после обработки.
        """
        with self._session() as s:
            q = (
                select(WebCommandRow)
                .where(WebCommandRow.consumed_at.is_(None))  # type: ignore[union-attr]
                .order_by(WebCommandRow.id.asc())
                .limit(limit)
            )
            return list(s.exec(q).all())

    def recent(self, limit: int = 50, consumed_only: bool = False) -> list[WebCommandRow]:
        """Последние N команд по issued_at desc — для history-view."""
        with self._session() as s:
            q = select(WebCommandRow).order_by(WebCommandRow.id.desc())
            if consumed_only:
                q = q.where(WebCommandRow.consumed_at.is_not(None))  # type: ignore[union-attr]
            q = q.limit(limit)
            return list(s.exec(q).all())

    def count_unconsumed(self) -> int:
        from sqlalchemy import func

        with self._session() as s:
            q = (
                select(func.count(WebCommandRow.id))
                .where(WebCommandRow.consumed_at.is_(None))  # type: ignore[union-attr]
            )
            return int(s.exec(q).one())

    def latest_pause_state(self, only_consumed: bool = True) -> bool:
        """Текущее состояние loop по последней pause/resume команде.

        True = paused, False = running (или нет ни одной pause-команды никогда).
        only_consumed=True: учитываем только консьюменные (фактически
        применённые agent run-loop). Это даёт survivable-state: после рестарта
        контейнера run-loop вернётся в pause если оператор раньше его поставил.
        """
        with self._session() as s:
            q = (
                select(WebCommandRow)
                .where(
                    WebCommandRow.command.in_(
                        [CommandType.PAUSE.value, CommandType.RESUME.value]
                    )
                )
                .order_by(WebCommandRow.id.desc())
                .limit(1)
            )
            if only_consumed:
                q = q.where(WebCommandRow.consumed_at.is_not(None))  # type: ignore[union-attr]
            rows = list(s.exec(q).all())
            if not rows:
                return False
            return rows[0].command == CommandType.PAUSE.value
