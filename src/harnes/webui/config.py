"""Webui-специфичный config — поверх harnes.config.Settings.

Webui переиспользует все пути из harnes.config (lancedb_path, goal_store,
journal_db_path и т.д.) — они read-only из контейнера агента через bind-mount.

Дополнительно — webui-сервер (host/port/reload).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebuiSettings(BaseSettings):
    """ENV-конфигурация webui-сервера.

    Все WEBUI_*-переменные читаются автоматически.
    Дефолты подходят для docker (порт 8000, бинд на 0.0.0.0).
    """

    model_config = SettingsConfigDict(
        env_prefix="WEBUI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "INFO"

    # Прятать кнопки управления (approve/reject/create) — read-only режим.
    read_only: bool = False

    # Путь до command-file (IPC с агентом). Пока зарезервировано для будущего.
    command_file: str = "/app/data/web_commands.jsonl"

    # HTTP Basic Auth. Если username пустой — auth disabled (loopback-default).
    # Использовать когда webui выйдет за 127.0.0.1 (reverse proxy, общий dev и т.п.).
    auth_username: str = ""
    auth_password: str = ""


_settings: WebuiSettings | None = None


def get_webui_settings() -> WebuiSettings:
    global _settings
    if _settings is None:
        _settings = WebuiSettings()
    return _settings
