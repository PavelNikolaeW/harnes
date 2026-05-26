"""Persistent eval-results — каждый прогон benchmark'а пишется в SQLite.

См. v0.3 #25, v1.0 #31+#32.

Хранилище: одна таблица eval_runs. Каждая строка — один прогон adapter'а
с агрегатными метриками + версионной telemetry (git_sha, skill_versions,
config_hash). Это база для eval-compare между prefix-точками разработки.

v1.0 #31 — поддержка held-out наборов (флаг held_out + idemta eval_set).
v1.0 #32 — reliability-first метрики (pass@k, stable@k, p50/p95, entropy).
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
from sqlalchemy import inspect, text
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

    # v1.0 #31 — held-out flag + eval-set identifier
    eval_set: str = Field(default="", index=True)
    eval_set_hash: str = ""  # хэш отсортированных task_id — для верификации
    held_out: bool = Field(default=False, index=True)

    # v1.0 #32 — reliability metrics
    repeat_k: int = 1
    pass_at_k: float = 0.0  # task succeeded at least once across k repeats
    stable_at_k: float = 0.0  # task gave same final answer across all k repeats
    p50_steps: float = 0.0
    p95_steps: float = 0.0
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    failure_entropy: float = 0.0  # Shannon entropy of failure_modes distribution


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


def compute_eval_set_hash(task_ids: list[str]) -> str:
    """Sha256[:16] от отсортированных task_id. Стабильно идентифицирует набор задач.

    Используется чтобы eval-compare ругался когда сравниваются прогоны с разным
    набором task'ов (даже при одинаковом eval_set метке).
    """
    sorted_ids = sorted(task_ids)
    blob = "|".join(sorted_ids).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------- Reliability helpers (v1.0 #32) ----------


def _quantile(data: list[float], p: float) -> float:
    """Linear-interpolated quantile. Без numpy. p ∈ [0, 1]."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _shannon_entropy(modes: dict[str, int]) -> float:
    """Shannon entropy (бит) над distribution failure_modes.

    Если все провалы один режим — 0. Если равномерно — log2(N_modes).
    """
    import math

    total = sum(modes.values())
    if total == 0:
        return 0.0
    h = -sum(
        (c / total) * math.log2(c / total)
        for c in modes.values()
        if c > 0
    )
    # Нормализуем -0.0 → 0.0 (IEEE 754) для cosmetic-вывода.
    return 0.0 if abs(h) < 1e-15 else h


def _canonical_answer(traj: Any) -> str:
    """Нормализованный финальный ответ траектории для stable@k сравнения."""
    if traj is None:
        return ""
    fs = getattr(traj, "final_state", None)
    if fs is None:
        return ""
    if isinstance(fs, dict):
        ans = fs.get("answer") or fs.get("response") or ""
    else:
        ans = fs
    return str(ans).strip().lower()


def compute_reliability_metrics(
    per_task: list[Any],
    repeat_k: int,
) -> dict[str, float]:
    """Считает reliability-метрики из per_task (PerTaskResult list).

    Группирует по task_id; пасс@k = доля task'ов, успешных хоть раз;
    stable@k = доля task'ов с одинаковым canonical answer по k запускам.

    p50/p95_steps, p50/p95_latency_s, failure_entropy — over всех per_task.
    """
    # Группируем per_task по task_id.
    by_task: dict[str, list[Any]] = {}
    for r in per_task:
        by_task.setdefault(r.task_id, []).append(r)

    if not by_task:
        return {
            "pass_at_k": 0.0,
            "stable_at_k": 0.0,
            "p50_steps": 0.0,
            "p95_steps": 0.0,
            "p50_latency_s": 0.0,
            "p95_latency_s": 0.0,
            "failure_entropy": 0.0,
        }

    n_tasks = len(by_task)
    pass_count = sum(1 for rs in by_task.values() if any(r.success for r in rs))

    # stable@k: для k=1 — каждая task тривиально стабильна.
    if repeat_k <= 1:
        stable_count = n_tasks
    else:
        stable_count = 0
        for rs in by_task.values():
            answers = {_canonical_answer(r.trajectory) for r in rs}
            if len(answers) == 1:
                stable_count += 1

    steps = [float(r.steps) for r in per_task]
    latencies = [float(getattr(r, "latency_s", 0.0) or 0.0) for r in per_task]

    failure_modes: dict[str, int] = {}
    for r in per_task:
        if not r.success and r.failure_mode:
            failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1

    return {
        "pass_at_k": pass_count / n_tasks,
        "stable_at_k": stable_count / n_tasks,
        "p50_steps": _quantile(steps, 0.5),
        "p95_steps": _quantile(steps, 0.95),
        "p50_latency_s": _quantile(latencies, 0.5),
        "p95_latency_s": _quantile(latencies, 0.95),
        "failure_entropy": _shannon_entropy(failure_modes),
    }


# ---------- Store ----------


class EvalHistoryStore:
    """SQLite-репозиторий прогонов benchmark'а."""

    # Новые поля v1.0 #31+#32 — мигрируются на старых БД через ALTER TABLE.
    _NEW_COLUMNS_DDL: dict[str, str] = {
        "eval_set": "TEXT NOT NULL DEFAULT ''",
        "eval_set_hash": "TEXT NOT NULL DEFAULT ''",
        "held_out": "INTEGER NOT NULL DEFAULT 0",  # SQLite BOOL = INTEGER
        "repeat_k": "INTEGER NOT NULL DEFAULT 1",
        "pass_at_k": "REAL NOT NULL DEFAULT 0.0",
        "stable_at_k": "REAL NOT NULL DEFAULT 0.0",
        "p50_steps": "REAL NOT NULL DEFAULT 0.0",
        "p95_steps": "REAL NOT NULL DEFAULT 0.0",
        "p50_latency_s": "REAL NOT NULL DEFAULT 0.0",
        "p95_latency_s": "REAL NOT NULL DEFAULT 0.0",
        "failure_entropy": "REAL NOT NULL DEFAULT 0.0",
    }

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        url = self._build_url(db_path)
        self.engine = create_engine(url, echo=False)
        SQLModel.metadata.create_all(self.engine)
        self._migrate_v1_0()

    @staticmethod
    def _build_url(db_path: Path | str) -> str:
        if db_path == ":memory:":
            return "sqlite:///:memory:"
        p = Path(db_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p}"

    def _migrate_v1_0(self) -> None:
        """Добавляет столбцы #31+#32 в старую БД (создана до v1.0).

        SQLModel create_all не делает ALTER TABLE для существующих таблиц, поэтому
        на pre-v1.0 БД новых колонок не будет. Идемпотентно: пропускаем то, что есть.
        """
        try:
            inspector = inspect(self.engine)
            if "eval_runs" not in inspector.get_table_names():
                return  # таблица только что создана — она уже с v1.0-схемой
            existing = {col["name"] for col in inspector.get_columns("eval_runs")}
            missing = [
                (col, ddl)
                for col, ddl in self._NEW_COLUMNS_DDL.items()
                if col not in existing
            ]
            if not missing:
                return
            with self.engine.begin() as conn:
                for col, ddl in missing:
                    conn.execute(text(f"ALTER TABLE eval_runs ADD COLUMN {col} {ddl}"))
            log.info("eval.history.migrated", added_columns=[c for c, _ in missing])
        except Exception as exc:  # noqa: BLE001
            log.warning("eval.history.migration_failed", error=str(exc))

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
        *,
        eval_set: str = "",
        eval_set_hash: str = "",
        held_out: bool = False,
        repeat_k: int = 1,
    ) -> EvalRunRecord:
        """Запись прогона.

        v1.0:
        - eval_set — короткий идентификатор набора задач ("mab_dev_ar", "mab_holdout_lru")
        - eval_set_hash — sha256 первых 16 hex по отсортированным task_id
        - held_out — флаг "не смотрим при разработке"
        - repeat_k — число запусков на task (для pass@k / stable@k)
        """
        metrics = compute_reliability_metrics(result.per_task, repeat_k)
        # Если hash не передан — вычисляем сами из per_task.
        if not eval_set_hash:
            eval_set_hash = compute_eval_set_hash(
                sorted({r.task_id for r in result.per_task})
            )
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
            eval_set=eval_set,
            eval_set_hash=eval_set_hash,
            held_out=held_out,
            repeat_k=repeat_k,
            pass_at_k=metrics["pass_at_k"],
            stable_at_k=metrics["stable_at_k"],
            p50_steps=metrics["p50_steps"],
            p95_steps=metrics["p95_steps"],
            p50_latency_s=metrics["p50_latency_s"],
            p95_latency_s=metrics["p95_latency_s"],
            failure_entropy=metrics["failure_entropy"],
        )
        with self._session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
        log.info(
            "eval.history.recorded",
            run_id=row.id,
            adapter=result.name,
            eval_set=eval_set,
            held_out=held_out,
            success_rate=result.success_rate,
            pass_at_k=row.pass_at_k,
            git_sha=row.git_sha[:8],
        )
        return row

    def get(self, run_id: int) -> EvalRunRecord | None:
        with self._session() as s:
            return s.get(EvalRunRecord, run_id)

    def list_runs(
        self,
        adapter_name: str | None = None,
        eval_set: str | None = None,
        include_held_out: bool = False,
        limit: int = 20,
    ) -> list[EvalRunRecord]:
        """v1.0: по дефолту фильтрует held_out (research-hygiene).

        Используй include_held_out=True для held-out релиз-замера.
        """
        with self._session() as s:
            q = select(EvalRunRecord).order_by(EvalRunRecord.id.desc())
            if adapter_name is not None:
                q = q.where(EvalRunRecord.adapter_name == adapter_name)
            if eval_set is not None:
                q = q.where(EvalRunRecord.eval_set == eval_set)
            if not include_held_out:
                q = q.where(EvalRunRecord.held_out == False)  # noqa: E712 — SQLAlchemy
            q = q.limit(limit)
            return list(s.exec(q).all())

    def latest(
        self,
        adapter_name: str | None = None,
        eval_set: str | None = None,
        include_held_out: bool = False,
    ) -> EvalRunRecord | None:
        rows = self.list_runs(
            adapter_name=adapter_name,
            eval_set=eval_set,
            include_held_out=include_held_out,
            limit=1,
        )
        return rows[0] if rows else None
