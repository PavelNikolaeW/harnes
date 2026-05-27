"""Episodic store on LanceDB.

См. `agent_architecture.html` § 13.

Три таблицы:
- trajectories: метаданные траекторий (id, goal_id, status, cost, time range)
- steps: типизированные шаги, content_json — JSON-сериализованный Step
- step_embeddings: vector-индекс для шагов (для recall по смыслу) — заполняется
  при write_trajectory, требует /v1/embeddings на роутере или fastembed-фолбэк.
  Шаги, у которых embed-call не удался / dim не совпал, попадают в БД БЕЗ
  embedding-row — recall fall-through на keyword/recency их подхватит.

Recall (см. MemoryRouter._recall_episodic): vector → keyword → recency, цепочка
fallback'ов. Vector работает, когда BGE-M3 endpoint (или совместимый) доступен.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import lancedb
import pyarrow as pa
import structlog

from harnes.react.schema import Cost, Step, Trajectory, TrajectoryStatus

log = structlog.get_logger()


TRAJECTORIES_TABLE = "trajectories"
STEPS_TABLE = "steps"
STEP_EMBEDDINGS_TABLE = "step_embeddings"

# Dim BGE-M3. Если включат другую модель — поменять и удалить старые embeddings.
EPISODIC_VECTOR_DIM = 1024


_STOPWORDS = frozenset(
    {
        # function words EN
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "at", "by", "for", "with", "from", "into",
        "and", "or", "but", "if", "not",
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "their",
        "this", "that", "these", "those",
        "do", "does", "did", "can", "could", "would", "should", "will",
        "may", "might", "have", "has", "had",
        "what", "where", "when", "why", "how", "who", "which",
        "as", "so", "than", "then",
        # function words RU
        "и", "или", "но", "не", "на", "в", "из", "от", "до", "по", "за", "о",
        "что", "это", "тот", "как", "так", "же", "бы", "ли", "уж",
    }
)


def extract_terms(query: str, max_terms: int = 8) -> list[str]:
    """Извлечь content-words из NL-запроса: lowercase, alnum, ≥3 chars, не стопворд.

    На «compute or multiplied» вернёт ['compute', 'multiplied']. На запрос из
    одних стопвордов («what is it») — пустой список (вызывающий уйдёт в recency).
    """
    tokens = re.findall(r"[a-zа-яё0-9]{3,}", query.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


_trajectory_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("goal_id", pa.string()),
        pa.field("parent_trajectory_id", pa.string()),
        pa.field("status", pa.string()),
        pa.field("total_cost_tokens", pa.int64()),
        pa.field("total_cost_latency", pa.float64()),
        pa.field("final_state_json", pa.string()),
        pa.field("started_at", pa.timestamp("us")),
        pa.field("ended_at", pa.timestamp("us")),
        pa.field("metadata_json", pa.string()),
    ]
)

_step_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("trajectory_id", pa.string()),
        pa.field("goal_id", pa.string()),
        pa.field("step_type", pa.string()),
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("cost_tokens", pa.int64()),
        pa.field("cost_latency", pa.float64()),
        pa.field("content_json", pa.string()),
    ]
)

_step_embeddings_schema = pa.schema(
    [
        pa.field("step_id", pa.string()),
        pa.field(
            "vector",
            pa.list_(pa.float32(), EPISODIC_VECTOR_DIM),
        ),
        pa.field("text", pa.string()),  # debug: что было embedded
        pa.field("timestamp", pa.timestamp("us")),  # для time-decay future
    ]
)


def _step_searchable_text(step: Step) -> str | None:
    """Что embed'ить для конкретного шага. None — шаг не индексируем.

    Эвристика: важные для recall шаги — thought/action/observation. PlanStep
    и CritiqueStep тоже могут быть полезны (cause-reasoning), но они редкие.
    RetryNote — мета-шум.
    """
    t = step.type
    if t == "thought":
        return getattr(step, "text", None) or None
    if t == "action":
        tool_id = getattr(step, "tool_id", "")
        args = getattr(step, "args", {})
        return f"{tool_id}({json.dumps(args)[:300]})"
    if t == "observation":
        payload = getattr(step, "payload", None)
        if payload:
            return json.dumps(payload)[:500]
        return getattr(step, "error_detail", None) or None
    if t == "plan":
        return getattr(step, "rationale", None) or None
    if t == "critique":
        return getattr(step, "reasoning", None) or None
    return None


class EpisodicStore:
    """LanceDB-обёртка для episodic-логирования траекторий."""

    def __init__(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(p))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        # NB: `table_names()` is deprecated в LanceDB но `list_tables()` в
        # этой версии возвращает unhashable-структуру; вернёмся к новому API,
        # когда оно стабилизируется.
        names = set(self.db.table_names())
        if TRAJECTORIES_TABLE not in names:
            self.db.create_table(TRAJECTORIES_TABLE, schema=_trajectory_schema)
        if STEPS_TABLE not in names:
            self.db.create_table(STEPS_TABLE, schema=_step_schema)
        if STEP_EMBEDDINGS_TABLE not in names:
            self.db.create_table(
                STEP_EMBEDDINGS_TABLE, schema=_step_embeddings_schema
            )

    # ---------- write ----------

    def write_trajectory(self, traj: Trajectory) -> None:
        """Сохраняет траекторию целиком: metadata + все шаги + embeddings.

        Embeddings (BGE-M3 1024-d по дефолту) — batch-один-вызов на trajectory,
        не per-step. При неудаче embed (server down, dim mismatch) — пишем без
        embeddings; recall fall-through на keyword/recency их подхватит.

        Идемпотентно по trajectory.id для steps (но не для metadata — повторная
        запись просто добавит дубликат, фильтрация по uniqueness — TBD).
        """
        meta = {
            "id": str(traj.id),
            "goal_id": str(traj.goal_id),
            "parent_trajectory_id": (
                str(traj.parent_trajectory_id) if traj.parent_trajectory_id else ""
            ),
            "status": traj.status.value if traj.status else "",
            "total_cost_tokens": traj.total_cost.tokens,
            "total_cost_latency": traj.total_cost.latency_seconds,
            "final_state_json": json.dumps(traj.final_state)
            if traj.final_state is not None
            else "",
            "started_at": traj.started_at.replace(tzinfo=None),
            "ended_at": traj.ended_at.replace(tzinfo=None) if traj.ended_at else None,
            "metadata_json": json.dumps(traj.metadata),
        }
        self.db.open_table(TRAJECTORIES_TABLE).add([meta])

        step_rows = [self._step_to_row(s, traj.id, traj.goal_id) for s in traj.steps]
        if step_rows:
            self.db.open_table(STEPS_TABLE).add(step_rows)

        # Embeddings: batch один вызов на trajectory, fail-soft.
        self._write_step_embeddings(traj.steps)

        log.debug(
            "episodic.trajectory.written",
            trajectory_id=str(traj.id),
            steps=len(traj.steps),
        )

    def _write_step_embeddings(self, steps: list[Step]) -> None:
        """Batch embed + write для всех индексируемых шагов trajectory.

        Fail-soft: при любой ошибке (LLM-сервер недоступен, dim mismatch,
        fastembed-fallback с другой размерностью) — log warning + skip.
        Шаги без embedding row просто не попадут в vector search — keyword
        / recency fallback их подхватит.
        """
        indexable: list[tuple[Step, str]] = []
        for s in steps:
            text = _step_searchable_text(s)
            if text:
                indexable.append((s, text))
        if not indexable:
            return

        try:
            from harnes.llm.embeddings import embed

            vectors = embed([text for _, text in indexable])
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "episodic.embeddings.embed_failed",
                steps=len(indexable),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return

        # Dim guard — fastembed fallback (768) под BGE-M3 schema (1024) уронит
        # PyArrow validation. Проверяем явно и skip при mismatch.
        if vectors and len(vectors[0]) != EPISODIC_VECTOR_DIM:
            log.warning(
                "episodic.embeddings.dim_mismatch",
                expected=EPISODIC_VECTOR_DIM,
                actual=len(vectors[0]),
                hint="вероятно fastembed fallback — schema требует BGE-M3 dim",
            )
            return

        rows = [
            {
                "step_id": str(s.id),
                "vector": v,
                "text": text[:1000],  # debug-friendly limit
                "timestamp": s.timestamp.replace(tzinfo=None),
            }
            for (s, text), v in zip(indexable, vectors)
        ]
        try:
            self.db.open_table(STEP_EMBEDDINGS_TABLE).add(rows)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "episodic.embeddings.write_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

    @staticmethod
    def _step_to_row(step: Step, trajectory_id: UUID, goal_id: UUID) -> dict[str, Any]:
        # `step` тут — это union одного из ThoughtStep/ActionStep/...; у всех
        # есть type/id/timestamp/cost, payload-поля живут на самом классе.
        content = step.model_dump(mode="json")
        return {
            "id": str(step.id),
            "trajectory_id": str(trajectory_id),
            "goal_id": str(goal_id),
            "step_type": step.type,
            "timestamp": step.timestamp.replace(tzinfo=None),
            "cost_tokens": step.cost.tokens,
            "cost_latency": step.cost.latency_seconds,
            "content_json": json.dumps(content),
        }

    # ---------- read ----------

    def get_trajectory_meta(self, trajectory_id: UUID) -> dict[str, Any] | None:
        df = (
            self.db.open_table(TRAJECTORIES_TABLE)
            .search()
            .where(f"id = '{trajectory_id}'")
            .limit(1)
            .to_list()
        )
        return df[0] if df else None

    def get_steps(self, trajectory_id: UUID) -> list[dict[str, Any]]:
        """Возвращает все шаги траектории в порядке timestamp."""
        rows = (
            self.db.open_table(STEPS_TABLE)
            .search()
            .where(f"trajectory_id = '{trajectory_id}'")
            .limit(10_000)
            .to_list()
        )
        rows.sort(key=lambda r: r["timestamp"])
        return rows

    def list_trajectories_for_goal(self, goal_id: UUID) -> list[dict[str, Any]]:
        return (
            self.db.open_table(TRAJECTORIES_TABLE)
            .search()
            .where(f"goal_id = '{goal_id}'")
            .limit(10_000)
            .to_list()
        )

    def recent_steps(self, limit: int = 100) -> list[dict[str, Any]]:
        """Последние N шагов across all trajectories."""
        rows = self.db.open_table(STEPS_TABLE).search().limit(limit * 10).to_list()
        rows.sort(key=lambda r: r["timestamp"], reverse=True)
        return rows[:limit]

    def search_steps_by_vector(
        self,
        query: str,
        limit: int = 10,
        distance_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """ANN-поиск по step_embeddings (BGE-M3). Возвращает шаги отсортированные
        по similarity (ближайшие первыми).

        Цепочка:
        1. embed(query) — может упасть на server-failure → fastembed (другой dim)
           → schema mismatch → ловим и возвращаем [].
        2. LanceDB vector search в step_embeddings.
        3. JOIN с steps по step_id — возвращаем step rows.

        distance_threshold — отсечь слишком далёкие результаты (L2 distance,
        чем меньше — тем ближе). None = без отсечения.

        Возвращает [], если ANN недоступен (нет embed-моделей в нужной dim,
        нет данных, ошибка). Caller должен fall-through на keyword/recency.
        """
        try:
            from harnes.llm.embeddings import embed

            vectors = embed([query])
        except Exception as exc:  # noqa: BLE001
            log.debug("episodic.search_vector.embed_failed", error=str(exc))
            return []
        if not vectors or len(vectors[0]) != EPISODIC_VECTOR_DIM:
            return []

        try:
            results = (
                self.db.open_table(STEP_EMBEDDINGS_TABLE)
                .search(vectors[0])
                .limit(limit * 2)  # запас перед JOIN — некоторые step_id могут быть осиротевшими
                .to_list()
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("episodic.search_vector.lance_failed", error=str(exc))
            return []
        if not results:
            return []

        # _distance — добавляется LanceDB при vector search; меньше = ближе.
        if distance_threshold is not None:
            results = [r for r in results if r.get("_distance", 1e9) <= distance_threshold]
            if not results:
                return []

        # Сохраняем порядок distance → загружаем steps в этом же порядке.
        step_ids = [r["step_id"] for r in results]
        if not step_ids:
            return []
        # LanceDB SQL IN — формируем quoted list.
        in_clause = ", ".join(f"'{sid}'" for sid in step_ids)
        step_rows = (
            self.db.open_table(STEPS_TABLE)
            .search()
            .where(f"id IN ({in_clause})")
            .limit(limit * 2)
            .to_list()
        )
        # Восстанавливаем порядок по step_ids (distance asc).
        by_id = {r["id"]: r for r in step_rows}
        ordered = [by_id[sid] for sid in step_ids if sid in by_id]
        return ordered[:limit]

    def search_steps_by_terms(
        self,
        terms: list[str],
        limit: int = 10,
        scan_limit: int = 5_000,
    ) -> list[dict[str, Any]]:
        """Keyword search по content_json. Возвращает шаги, у которых content
        содержит хотя бы один term (case-insensitive), отсортированные
        (term-hits desc, timestamp desc).

        Не использует LanceDB filter DSL — простой scan + Python scoring; для
        десятков-сотен тысяч шагов хватит, дальше нужен FTS/ANN (см. follow-up).
        Возвращает [], если terms пуст.
        """
        if not terms:
            return []
        terms_lower = [t.lower() for t in terms if t]
        rows = (
            self.db.open_table(STEPS_TABLE).search().limit(scan_limit).to_list()
        )
        scored: list[tuple[int, datetime, dict[str, Any]]] = []
        for r in rows:
            c = r["content_json"].lower()
            hits = sum(1 for t in terms_lower if t in c)
            if hits > 0:
                scored.append((hits, r["timestamp"], r))
        # stable sort: by timestamp desc, then by hits desc → final order is
        # (hits desc, ts desc) since Python sort is stable.
        scored.sort(key=lambda x: x[1], reverse=True)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, _, r in scored[:limit]]

    def recent_trajectories(
        self, limit: int = 20, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Последние N трейекторий (по started_at desc), опционально по статусу."""
        q = self.db.open_table(TRAJECTORIES_TABLE).search().limit(limit * 10)
        if status is not None:
            q = q.where(f"status = '{status}'")
        rows = q.to_list()
        rows.sort(key=lambda r: r["started_at"], reverse=True)
        return rows[:limit]
