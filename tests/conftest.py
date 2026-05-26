"""Pytest fixtures and configuration."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_settings_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the cached Settings singleton before each test for isolation.

    Tests могут мутировать settings.* поля или env-переменные; перед каждым
    тестом форсим перезагрузку, чтобы изменения не утекали между тестами.
    """
    monkeypatch.setattr("harnes.config._settings", None)
