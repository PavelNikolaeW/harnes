"""Tests for v1.0 #35: TickJournal — событийный лог метацикла."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harnes.metacycle.journal import TickEventType, TickJournal


def test_append_and_query(tmp_path: Path) -> None:
    j = TickJournal(tmp_path / "j.db")
    j.append(0, TickEventType.LOOP_STARTED, {"interval": 5.0})
    j.append(1, TickEventType.TICK_STARTED)
    j.append(1, TickEventType.TICK_DONE, {"verdict": "success"})

    events = j.recent_events(limit=10)
    assert len(events) == 3
    # recent = по убыванию id
    assert events[0].event_type == TickEventType.TICK_DONE.value


def test_filter_by_event_type(tmp_path: Path) -> None:
    j = TickJournal(tmp_path / "j.db")
    j.append(0, TickEventType.TICK_STARTED)
    j.append(0, TickEventType.GOAL_SPAWNED, {"goal_id": "g1"})
    j.append(1, TickEventType.GOAL_SPAWNED, {"goal_id": "g2"})
    j.append(2, TickEventType.TICK_IDLE)

    spawned = j.recent_events(event_type=TickEventType.GOAL_SPAWNED)
    assert len(spawned) == 2
    assert all(e.event_type == TickEventType.GOAL_SPAWNED.value for e in spawned)


def test_filter_by_tick_id(tmp_path: Path) -> None:
    j = TickJournal(tmp_path / "j.db")
    for tick in range(3):
        j.append(tick, TickEventType.TICK_STARTED)
        j.append(tick, TickEventType.TICK_DONE)

    tick_1 = j.recent_events(tick_id=1, limit=10)
    assert len(tick_1) == 2
    assert all(e.tick_id == 1 for e in tick_1)


def test_snapshot_and_latest(tmp_path: Path) -> None:
    j = TickJournal(tmp_path / "j.db")
    assert j.latest_snapshot() is None

    s1 = j.snapshot(
        tick_id=10,
        processed_count=5,
        idle_count=5,
        ticks_with_self_spawn=2,
        total_self_spawned=3,
    )
    s2 = j.snapshot(
        tick_id=50,
        processed_count=20,
        idle_count=30,
        error_count=1,
        ticks_with_self_spawn=8,
        total_self_spawned=15,
    )

    assert s1 is not None
    assert s2 is not None

    latest = j.latest_snapshot()
    assert latest is not None
    assert latest.tick_id == 50
    assert latest.processed_count == 20
    assert latest.error_count == 1
    assert latest.total_self_spawned == 15


def test_stats(tmp_path: Path) -> None:
    j = TickJournal(tmp_path / "j.db")
    j.append(0, TickEventType.LOOP_STARTED)
    j.append(1, TickEventType.TICK_DONE)
    j.append(2, TickEventType.TICK_DONE)
    j.append(3, TickEventType.GOAL_SPAWNED)
    j.snapshot(tick_id=3, processed_count=2, idle_count=1)

    stats = j.stats()
    assert stats["total_events"] == 4
    assert stats["total_snapshots"] == 1
    assert stats["by_event_type"]["tick_done"] == 2
    assert stats["by_event_type"]["goal_spawned"] == 1
    assert stats["min_tick_id"] == 0
    assert stats["max_tick_id"] == 3


def test_event_count(tmp_path: Path) -> None:
    j = TickJournal(tmp_path / "j.db")
    j.append(0, TickEventType.TICK_DONE)
    j.append(1, TickEventType.TICK_DONE)
    j.append(2, TickEventType.GOAL_SPAWNED)

    assert j.event_count() == 3
    assert j.event_count(event_type=TickEventType.TICK_DONE) == 2
    assert j.event_count(event_type=TickEventType.GOAL_SPAWNED) == 1


def test_persists_to_disk(tmp_path: Path) -> None:
    db_path = tmp_path / "persistent.db"
    j1 = TickJournal(db_path)
    j1.append(0, TickEventType.TICK_DONE, {"hi": "world"})
    j1.snapshot(tick_id=0, processed_count=1, idle_count=0)

    j2 = TickJournal(db_path)
    events = j2.recent_events()
    assert len(events) == 1
    snap = j2.latest_snapshot()
    assert snap is not None
    assert snap.processed_count == 1


def test_recent_events_since(tmp_path: Path) -> None:
    j = TickJournal(tmp_path / "j.db")
    j.append(0, TickEventType.TICK_DONE)
    later = datetime.now(UTC) + timedelta(hours=1)

    events_since_future = j.recent_events(since=later)
    assert events_since_future == []
    events_since_past = j.recent_events(
        since=datetime.now(UTC) - timedelta(hours=1)
    )
    assert len(events_since_past) == 1


# ---------- Integration: run-loop с journal ----------


def test_run_loop_journal_integration(tmp_path: Path, monkeypatch) -> None:
    """run-loop stub-режим с journal записывает события и итоговый snapshot."""
    from click.testing import CliRunner
    from harnes.operator.cli import cli

    journal_path = tmp_path / "loop_journal.db"

    monkeypatch.setenv("EVAL__HISTORY_DB_PATH", str(tmp_path / "eval.db"))
    monkeypatch.setenv("MEMORY__LANCEDB_PATH", str(tmp_path / "lance"))
    monkeypatch.setenv("MEMORY__NEO4J_URI", "bolt://nowhere.invalid:7687")
    monkeypatch.setenv("GOAL_STORE__SQLITE_PATH", str(tmp_path / "goals.db"))
    monkeypatch.setenv("METACYCLE__JOURNAL_DB_PATH", str(journal_path))
    monkeypatch.setenv("METACYCLE__SNAPSHOT_EVERY_TICKS", "2")

    # Force singleton reset (conftest autouse делает это, но мы внутри Click runner).
    import harnes.config
    harnes.config._settings = None

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-loop",
            "--stub",
            "--max-ticks",
            "4",
            "--interval",
            "0",
            "--no-world",
            "--no-resume",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    # Журнал должен содержать LOOP_STARTED + N x TICK_STARTED + ... + LOOP_STOPPED
    journal = TickJournal(journal_path)
    stats = journal.stats()
    assert stats["total_events"] > 0
    assert stats["by_event_type"].get("loop_started", 0) >= 1
    assert stats["by_event_type"].get("loop_stopped", 0) >= 1
    assert stats["by_event_type"].get("tick_started", 0) >= 1
    # snapshot хотя бы один (snapshot_every=2 × max_ticks=4 → 2 промежуточных +
    # финальный в finally).
    assert stats["total_snapshots"] >= 1


def test_run_loop_resume_continues_tick_id(tmp_path: Path, monkeypatch) -> None:
    """Второй запуск run-loop с --resume стартует от tick_id последнего snapshot + 1."""
    from click.testing import CliRunner
    from harnes.operator.cli import cli

    journal_path = tmp_path / "loop_journal.db"

    common_env = {
        "EVAL__HISTORY_DB_PATH": str(tmp_path / "eval.db"),
        "MEMORY__LANCEDB_PATH": str(tmp_path / "lance"),
        "MEMORY__NEO4J_URI": "bolt://nowhere.invalid:7687",
        "GOAL_STORE__SQLITE_PATH": str(tmp_path / "goals.db"),
        "METACYCLE__JOURNAL_DB_PATH": str(journal_path),
        "METACYCLE__SNAPSHOT_EVERY_TICKS": "1",
    }
    for k, v in common_env.items():
        monkeypatch.setenv(k, v)

    import harnes.config
    harnes.config._settings = None

    runner = CliRunner()

    # 1-й прогон: 3 тика.
    runner.invoke(
        cli,
        [
            "run-loop", "--stub", "--max-ticks", "3", "--interval", "0",
            "--no-world", "--no-resume",
        ],
        catch_exceptions=False,
    )

    # 2-й прогон: 3 тика, --resume.
    harnes.config._settings = None  # reset config singleton
    result2 = runner.invoke(
        cli,
        [
            "run-loop", "--stub", "--max-ticks", "6", "--interval", "0",
            "--no-world",  # default resume=True
        ],
        catch_exceptions=False,
    )
    assert result2.exit_code == 0, result2.output
    assert "Resuming from snapshot" in result2.output
