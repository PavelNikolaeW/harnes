"""Runtime configuration.

Source precedence (high to low):
  1. Environment variables (nested via `__`, e.g. LLM__MODEL=foo)
  2. config/default.yaml (or path passed to Settings.load)
  3. Built-in defaults on each model class

Loaded once at process start via `get_settings()`; subsequent calls return
the cached singleton.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


def _default_tiers() -> dict[str, str]:
    """В dev все тиры мапятся на gemma-26b-a4b (быстрая, MoE 4B active).

    В проде поднимем gemma-31b-mtp для main, qwen-35b для heavy.
    """
    return {
        "light": "gemma-26b-a4b",
        "main": "gemma-26b-a4b",
        "heavy": "gemma-26b-a4b",
    }


class LLMConfig(BaseModel):
    api_base: str = "http://192.168.0.111:8000/v1"
    api_key: str = "dummy"
    # Дефолтная модель — используется если tier не указан явно (backward-compat).
    model: str = "gemma-26b-a4b"
    timeout: int = 60
    max_retries: int = 3
    # Tier-абстракция. light для attend/critic/verify, main для thought/action,
    # heavy для reflect. В dev все три на одной модели.
    tiers: dict[str, str] = Field(default_factory=_default_tiers)
    default_tier: str = "main"


class MemoryConfig(BaseModel):
    lancedb_path: Path = Path("/app/data/lancedb")
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "harnes-dev-pw"


class GoalStoreConfig(BaseModel):
    sqlite_path: Path = Path("/app/data/goals.db")


class ProceduralStoreConfig(BaseModel):
    sqlite_path: Path = Path("/app/data/skill_metrics.db")
    bundles_dir: Path = Path("/app/skills")


class EmbeddingsConfig(BaseModel):
    use_server: bool = False
    model: str = "BAAI/bge-m3"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trace_log_path: Path = Path("/app/data/traces")


class TickConfig(BaseModel):
    min_interval_seconds: float = 1.0
    budget_default_tokens: int = 50_000


class EvalConfig(BaseModel):
    """Eval-results persistence (см. v0.3 #25)."""

    history_db_path: Path = Path("/app/data/eval_history.db")


class Settings(BaseSettings):
    """Root settings object.

    Precedence (high → low):
      1. Environment variables (`MODEL__FIELD` nested)
      2. YAML файл (config/default.yaml или /app/config/default.yaml)
      3. Built-in defaults на каждом BaseModel-классе
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    goal_store: GoalStoreConfig = GoalStoreConfig()
    procedural_store: ProceduralStoreConfig = ProceduralStoreConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    logging: LoggingConfig = LoggingConfig()
    tick: TickConfig = TickConfig()
    eval: EvalConfig = EvalConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env > YAML (init kwargs) > dotenv > defaults.

        Без этой кастомизации pydantic-settings ставит init_settings первым,
        и YAML-значения из cls(**yaml_data) перевешивают env-переменные.
        """
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings. YAML грузится как init kwargs, env-переменные перевешивают."""
        if config_path is None:
            for candidate in (
                Path("/app/config/default.yaml"),
                Path("config/default.yaml"),
            ):
                if candidate.exists():
                    config_path = candidate
                    break

        if config_path is not None and config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            return cls(**yaml_data)
        return cls()


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached singleton, loading it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
