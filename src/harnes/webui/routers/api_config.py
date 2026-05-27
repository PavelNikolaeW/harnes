"""Live config viewer — read-only render всего Settings-tree + env-override attribution.

Оператору удобно видеть какие настройки сейчас активны без `cat config/default.yaml`
+ `printenv | grep`. Изменения требуют рестарта контейнера — это здесь только окно
во внутренности агента.

Источники, которые сводятся вместе:
  1. `Settings.model_dump(mode='json')` — итоговый снимок (env > yaml > defaults).
  2. `os.environ` — фильтр по prefix'ам секций (`LLM__`, `MEMORY__`, ...). Если
     env-var есть и его имя матчит секцию/ключ — он считается источником.
  3. Какой config YAML был фактически загружен (см. `Settings.load`).

Секреты (`api_key`, `password`, etc.) маскируются точечно по имени поля — потенциальные
ложно-положительные срабатывания меньше боли, чем утечка пароля Neo4j на экран.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from harnes import AGENT_NAME
from harnes.config import Settings
from harnes.webui.config import WebuiSettings, get_webui_settings
from harnes.webui.deps import get_agent_settings
from harnes.webui.templating import templates

router = APIRouter()


# Поля, значения которых не показываем в открытом виде. Матчинг — точечно по
# имени field'а (не по содержимому): меньше false-positives, проще аудит. Если
# схема расширится — пополнять здесь.
_SECRET_KEYS: tuple[str, ...] = (
    "api_key",
    "password",
    "neo4j_password",
    "secret",
)

_SECRET_MASK = "••• (hidden)"

# Кандидаты config-file (порядок == precedence в Settings.load).
_CONFIG_PATH_CANDIDATES: tuple[Path, ...] = (
    Path("/app/config/default.yaml"),
    Path("config/default.yaml"),
)


def _is_secret_key(key: str) -> bool:
    return key.lower() in _SECRET_KEYS


def _mask(value: Any, key: str) -> Any:
    """Заменяет значение секретного поля на маску. Не-секреты возвращает as-is."""
    if _is_secret_key(key) and value not in (None, ""):
        return _SECRET_MASK
    return value


def _resolve_config_file() -> str:
    """Возвращает строковый путь к загруженному YAML, или '—' если ни один не существует.

    Логика повторяет `Settings.load` — берём первый существующий из кандидатов.
    Если YAML не загружался (только defaults + env), возвращаем '—'.
    """
    for p in _CONFIG_PATH_CANDIDATES:
        if p.exists():
            return str(p.resolve())
    return "—"


def _collect_env_overrides(section_names: list[str]) -> dict[str, str]:
    """Собирает env-vars, начинающиеся с {SECTION}__.

    pydantic-settings парсит `LLM__MODEL=foo` → `settings.llm.model`, разделитель
    `__` (см. `Settings.model_config.env_nested_delimiter`). Здесь — обратный шаг:
    из process env выбираем переменные с известными префиксами, чтобы потом
    атрибутировать конкретным полям settings tree.

    Возвращает dict {ENV_VAR_NAME: env_value}. Значения секретов НЕ маскируются
    здесь — маскировка делается на render-уровне через `_mask`.
    """
    prefixes = tuple(f"{name.upper()}__" for name in section_names)
    return {k: v for k, v in os.environ.items() if k.startswith(prefixes)}


def _env_var_for(section: str, key_path: list[str]) -> str:
    """Имя env-var, которое в pydantic-settings переопределяет это поле.

    `_env_var_for("llm", ["model"])` → `"LLM__MODEL"`.
    `_env_var_for("memory", ["lancedb_path"])` → `"MEMORY__LANCEDB_PATH"`.
    Для nested-dict (`tiers.light`) — `LLM__TIERS__LIGHT`.
    """
    parts = [section.upper()] + [p.upper() for p in key_path]
    return "__".join(parts)


def _flatten_section(
    section_name: str,
    section_dict: dict[str, Any],
    env_overrides: dict[str, str],
) -> list[dict[str, Any]]:
    """Превращает секцию (потенциально вложенную) в плоский список row-dicts.

    Каждый row = {key, value, env_var, env_override_value (если есть)}.

    Nested dict (например `llm.tiers = {"light": "foo", "main": "bar"}`) разворачивается
    в отдельные строки с key = "tiers.light", "tiers.main". Это позволяет показать
    env-override на уровне конкретного nested-ключа (LLM__TIERS__LIGHT).
    """
    rows: list[dict[str, Any]] = []

    def _walk(prefix: list[str], value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _walk([*prefix, k], v)
            return

        key_display = ".".join(prefix)
        leaf_key = prefix[-1] if prefix else ""
        env_var = _env_var_for(section_name, prefix)
        env_present = env_var in env_overrides
        rows.append(
            {
                "key": key_display,
                "value": _mask(value, leaf_key),
                "env_var": env_var if env_present else None,
                "env_value": (
                    _mask(env_overrides[env_var], leaf_key) if env_present else None
                ),
            }
        )

    for key, value in section_dict.items():
        _walk([key], value)

    return rows


def _webui_dump(webui_cfg: WebuiSettings) -> list[dict[str, Any]]:
    """Плоский dump WebuiSettings для small-table в подвале."""
    dump = webui_cfg.model_dump(mode="json")
    rows: list[dict[str, Any]] = []
    for key, value in dump.items():
        env_var = f"WEBUI_{key.upper()}"
        env_present = env_var in os.environ
        rows.append(
            {
                "key": key,
                "value": _mask(value, key),
                "env_var": env_var if env_present else None,
                "env_value": (
                    _mask(os.environ[env_var], key) if env_present else None
                ),
            }
        )
    return rows


@router.get("", response_class=HTMLResponse)
def show_config(
    request: Request,
    settings: Settings = Depends(get_agent_settings),
    webui_cfg: WebuiSettings = Depends(get_webui_settings),
) -> HTMLResponse:
    """Read-only страница с full settings tree + env-source attribution."""
    dump = settings.model_dump(mode="json")

    section_names = list(dump.keys())
    env_overrides = _collect_env_overrides(section_names)

    sections: list[dict[str, Any]] = []
    for name in section_names:
        section_dict = dump[name]
        if not isinstance(section_dict, dict):
            # Top-level scalar — оборачиваем для единообразия с table.
            section_dict = {"value": section_dict}
        rows = _flatten_section(name, section_dict, env_overrides)
        sections.append(
            {
                "name": name,
                "rows": rows,
                "env_override_count": sum(1 for r in rows if r["env_var"]),
            }
        )

    overview = {
        "agent_name": AGENT_NAME,
        "python_version": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "config_file": _resolve_config_file(),
        "env_overrides_total": len(env_overrides),
    }

    return templates.TemplateResponse(
        request,
        "config/show.html",
        {
            "overview": overview,
            "sections": sections,
            "webui_rows": _webui_dump(webui_cfg),
        },
    )
