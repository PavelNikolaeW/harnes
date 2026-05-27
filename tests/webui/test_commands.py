"""Tests для CommandStore + webui /commands/pause|resume|trigger.

Покрытие:
- unit: issue/drain/mark_consumed/latest_pause_state цикл.
- behaviour: webui-POST → команда в store.
- security: WEBUI_READ_ONLY=true → 403 на POST.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harnes.metacycle.commands import (
    CommandStore,
    CommandType,
    ConsumedStatus,
)

# ---------- unit: CommandStore ----------


def test_command_store_issue_drain(tmp_data_dir: Path) -> None:
    """issue pause → drain returns 1 → mark_consumed → drain empty → state True."""
    store = CommandStore(tmp_data_dir / "cmds_basic.db")
    row = store.issue(CommandType.PAUSE)
    assert row.id is not None

    drained = store.drain()
    assert len(drained) == 1
    assert drained[0].command == CommandType.PAUSE.value
    assert drained[0].consumed_at is None

    store.mark_consumed(row.id, ConsumedStatus.OK)
    assert store.drain() == []
    assert store.latest_pause_state(only_consumed=True) is True


def test_command_store_latest_pause_state_unconsumed(tmp_data_dir: Path) -> None:
    """Issue pause без consume: only_consumed=True → False, only_consumed=False → True."""
    store = CommandStore(tmp_data_dir / "cmds_unconsumed.db")
    store.issue(CommandType.PAUSE)
    assert store.latest_pause_state(only_consumed=True) is False
    assert store.latest_pause_state(only_consumed=False) is True


def test_pause_resume_state(tmp_data_dir: Path) -> None:
    """pause+consumed → True; затем resume+consumed → False."""
    store = CommandStore(tmp_data_dir / "cmds_pauseresume.db")
    pause = store.issue(CommandType.PAUSE)
    store.mark_consumed(pause.id, ConsumedStatus.OK)
    assert store.latest_pause_state(only_consumed=True) is True

    resume = store.issue(CommandType.RESUME)
    store.mark_consumed(resume.id, ConsumedStatus.OK)
    assert store.latest_pause_state(only_consumed=True) is False


# ---------- behaviour: webui POST ----------


def test_webui_pause_post(client: TestClient, webui_env: Path) -> None:
    """POST /commands/pause → 303; store содержит 1 unconsumed pause."""
    r = client.post("/commands/pause", follow_redirects=False)
    assert r.status_code == 303

    # Открываем тот же store, в который писало приложение.
    from harnes.config import get_settings

    settings = get_settings()
    store = CommandStore(settings.metacycle.commands_db_path)
    pending = store.drain()
    assert len(pending) == 1
    assert pending[0].command == CommandType.PAUSE.value
    assert pending[0].consumed_at is None


# ---------- security: read-only ----------


def test_webui_read_only_blocks_post(
    monkeypatch: pytest.MonkeyPatch, webui_env: Path
) -> None:
    """WEBUI_READ_ONLY=true → POST /commands/pause → 403.

    Read-only state хранится в кэшированном WebuiSettings. Сбрасываем
    singleton и пересоздаём app с новым env.
    """
    monkeypatch.setenv("WEBUI_READ_ONLY", "true")
    monkeypatch.setattr("harnes.webui.config._settings", None)

    from harnes.webui.app import create_app

    app = create_app()
    with TestClient(app) as c:
        r = c.post("/commands/pause", follow_redirects=False)
        assert r.status_code == 403
