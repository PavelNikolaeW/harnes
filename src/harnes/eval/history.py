"""Persistent eval-results — каждый прогон benchmark'а пишется в SQLite.

См. v0.3 #25.

Хранилище: одна таблица eval_runs. Каждая строка — один прогон adapter'а
с агрегатными метриками + версионной telemetry (git_sha, skill_versions,
config_hash). Это база для eval-compare между prefix-точками разработки.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import structlog
from sqlmodel import Field, Session, SQLModel, create_engine, select

from harnes.config import get_settings
from harnes.eval.harness import EvalResult

log = structlog.get_logger()


# ---------- Row ----------


class EvalRunRecord(SQLModel, table=True):
    """Одна запись прогона benchmark'а."""

    __tablename__ = "eval_runs"

    id: int | None = Field(default=None, primary_key=True)
    adapter_name: str = Field(index=True)
    started_at: datetime
    ended_at: datetime

    total_tasks: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    avg_steps: float = 0.0
    avg_cost_tokens: float = 0.0

    failure_modes_json: str = "{}"  # JSON {mode: count}
    git_sha: str = ""
    skill_versions_json: str = "{}"  # {skill_id: version}
    config_hash: str = ""
    notes: str = ""


# ---------- Snapshot helpers ----------


def _current_git_sha() -> str:
    """Возвращает текущий HEAD SHA (или пустую строку если git недоступен)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _capture_config_hash() -> str:
    """Hash relevant config полей: model/tiers/timeout/max_retries/budgets."""
    settings = get_settings()
    relevant = {
        "llm_model": settings.llm.model,
        "llm_tiers": dict(settings.llm.tiers),
        "llm_timeout": settings.llm.timeout,
        "llm_max_retries": settings.llm.max_retries,
        "tick_budget_default_tokens": settings.tick.budget_default_tokens,
    }
    blob = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def capture_skill_versions(skill_registry: Any) -> dict[str, str]:
    """Snapshot {skill_id: version} всех скиллов в registry."""
    try:
        return {s.id: s.version for s in skill_registry.load_all()}
    except Exception as exc:  # noqa: BLE001
        log.warning("eval.capture_skill_versions.failed", error=str(exc))
        return {}


# ---------- Store ----------


class EvalHistoryStore:
    """SQLite-репозиторий прогонов benchmark'а."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        url = self._build_url(db_path)
        self.engine = create_engine(url, echo=False)
        SQLModel.metadata.create_all(self.engine)

    @staticmethod
    def _build_url(db_path: Path | str) -> str:
        if db_path == ":memory:":
            return "sqlite:///:memory:"
        p = Path(db_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p}"

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self.engine) as s:
            yield s

    def record(
        self,
        result: EvalResult,
        started_at: datetime,
        ended_at: datetime,
        skill_versions: dict[str, str] | None = None,
        notes: str = "",
    ) -> EvalRunRecord:
        row = EvalRunRecord(
            adapter_name=result.name,
            started_at=started_at,
            ended_at=ended_at,
            total_tasks=len(result.per_task),
            success_count=sum(1 for r in result.per_task if r.success),
            success_rate=result.success_rate,
            avg_steps=result.avg_steps,
            avg_cost_tokens=result.avg_cost_tokens,
            failure_modes_json=json.dumps(result.failure_modes),
            git_sha=_current_git_sha(),
            skill_versions_json=json.dumps(skill_versions or {}),
            config_hash=_capture_config_hash(),
            notes=notes,
        )
        with self._session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
        log.info(
            "eval.history.recorded",
            run_id=row.id,
            adapter=result.name,
            success_rate=result.success_rate,
            git_sha=row.git_sha[:8],
        )
        return row

    def get(self, run_id: int) -> EvalRunRecord | None:
        with self._session() as s:
            return s.get(EvalRunRecord, run_id)

    def list_runs(
        self,
        adapter_name: str | None = None,
        limit: int = 20,
    ) -> list[EvalRunRecord]:
        with self._session() as s:
            q = select(EvalRunRecord).order_by(EvalRunRecord.id.desc())
            if adapter_name is not None:
                q = q.where(EvalRunRecord.adapter_name == adapter_name)
            q = q.limit(limit)
            return list(s.exec(q).all())

    def latest(self, adapter_name: str | None = None) -> EvalRunRecord | None:
        rows = self.list_runs(adapter_name=adapter_name, limit=1)
        return rows[0] if rows else None
