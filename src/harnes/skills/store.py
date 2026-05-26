"""Skill registry — git-versioned YAML bundles + SQLite metrics.

См. `agent_architecture.html` § 9.

Архитектура:
- Бандлы скиллов — YAML-файлы в `bundles_dir` (один файл — один скилл).
  Файлы коммитятся в git; версионирование = git tags.
- Метрики — SQLite-таблица invocations, агрегируется on demand в SkillMetrics.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import structlog
import yaml
from sqlmodel import Field, Session, SQLModel, create_engine, select

from harnes.goals.schema import GoalClass
from harnes.skills.schema import Skill, SkillMetrics, SkillStatus

log = structlog.get_logger()


# ---------- Invocation row ----------


class InvocationRow(SQLModel, table=True):
    """Один эпизод вызова скилла. Метрики агрегируются по этой таблице."""

    __tablename__ = "skill_invocations"

    id: int | None = Field(default=None, primary_key=True)
    skill_id: str = Field(index=True)
    skill_version: str
    timestamp: datetime
    success: bool
    cost_tokens: int = 0
    steps: int = 0
    failure_mode: str | None = None
    warning: bool = False


# ---------- Registry ----------


class SkillRegistry:
    """Чтение бандлов из файловой системы + метрики per-version в SQLite."""

    def __init__(
        self,
        bundles_dir: Path | str,
        metrics_db: Path | str = ":memory:",
    ) -> None:
        self.bundles_dir = Path(bundles_dir)
        self.bundles_dir.mkdir(parents=True, exist_ok=True)

        url = (
            "sqlite:///:memory:"
            if metrics_db == ":memory:"
            else f"sqlite:///{Path(metrics_db).resolve()}"
        )
        self.engine = create_engine(url, echo=False)
        SQLModel.metadata.create_all(self.engine)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self.engine) as s:
            yield s

    # ---------- Bundle loading ----------

    def _load_yaml(self, path: Path) -> Skill:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Skill.model_validate(data)

    def load_all(self) -> list[Skill]:
        skills: list[Skill] = []
        for path in sorted(self.bundles_dir.glob("*.yaml")):
            try:
                skills.append(self._load_yaml(path))
            except Exception as exc:  # noqa: BLE001 — log and skip bad file
                log.error(
                    "skill.bundle.load_failed",
                    path=str(path),
                    error=str(exc),
                )
        return skills

    def get(self, skill_id: str) -> Skill | None:
        path = self.bundles_dir / f"{skill_id}.yaml"
        if not path.exists():
            return None
        return self._load_yaml(path)

    def list_active(self) -> list[Skill]:
        return [s for s in self.load_all() if s.status == SkillStatus.ACTIVE]

    def list_applicable(self, goal_class: GoalClass) -> list[Skill]:
        """Скиллы, применимые к данному классу цели.

        Если у скилла `applicable_goal_classes` пустой — считаем универсальным.
        """
        result = []
        for skill in self.list_active():
            if (
                not skill.applicable_goal_classes
                or goal_class in skill.applicable_goal_classes
            ):
                result.append(skill)
        return result

    # ---------- Bundle write ----------

    def save(self, skill: Skill) -> None:
        """Запись бандла в YAML. Используется при создании скилла оператором
        или (позже) reflect'ом при рефайнах. Версионирование — через git.
        """
        path = self.bundles_dir / f"{skill.id}.yaml"
        data = skill.model_dump(mode="json", exclude_none=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        log.info("skill.bundle.saved", skill_id=skill.id, path=str(path))

    # ---------- Metrics ----------

    def record_invocation(
        self,
        skill_id: str,
        skill_version: str,
        success: bool,
        cost_tokens: int = 0,
        steps: int = 0,
        failure_mode: str | None = None,
        warning: bool = False,
    ) -> None:
        with self._session() as s:
            s.add(
                InvocationRow(
                    skill_id=skill_id,
                    skill_version=skill_version,
                    timestamp=datetime.now(UTC),
                    success=success,
                    cost_tokens=cost_tokens,
                    steps=steps,
                    failure_mode=failure_mode,
                    warning=warning,
                )
            )
            s.commit()

    def get_metrics(
        self,
        skill_id: str,
        version: str | None = None,
    ) -> SkillMetrics:
        """Агрегированные метрики по invocations. Если version=None — по всем версиям."""
        with self._session() as s:
            query = select(InvocationRow).where(InvocationRow.skill_id == skill_id)
            if version is not None:
                query = query.where(InvocationRow.skill_version == version)
            rows = list(s.exec(query).all())

        if not rows:
            return SkillMetrics()

        n = len(rows)
        success_count = sum(1 for r in rows if r.success)
        failure_modes: dict[str, int] = {}
        for r in rows:
            if not r.success and r.failure_mode:
                failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1
        warning_count = sum(1 for r in rows if r.warning)

        return SkillMetrics(
            invocation_count=n,
            success_rate=success_count / n,
            avg_cost_tokens=sum(r.cost_tokens for r in rows) / n,
            avg_steps=sum(r.steps for r in rows) / n,
            failure_modes=failure_modes,
            warning_rate=warning_count / n,
        )
