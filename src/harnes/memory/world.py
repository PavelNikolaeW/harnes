"""World model store — temporal KG on Graphiti + Neo4j.

См. `agent_architecture.html` § 14.

В v0 это **stub-обёртка**: создаёт клиент, имеет API для add_episode и search,
но детальная реализация (схема узлов, политика разрешения конфликтов,
калибровка предсказаний) — отдельная задача после v0 (см. open questions).

Реальная интеграция с Graphiti добавится, когда world_update этап начнёт
обновлять модель мира. Сейчас MemoryRouter всё равно умеет работать без world.
"""
from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


class WorldModelStore:
    """Stub-обёртка над Graphiti. v0: операции no-op, search возвращает пусто."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ) -> None:
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self._client: Any = None  # lazy

    def _connect(self) -> None:
        """Реальная инициализация Graphiti — отложена до первого вызова.

        В v0 stub'е этот метод просто отмечает «подключения нет, операции no-op».
        Когда подключаем Graphiti — здесь импорт graphiti_core и async-конструктор.
        """
        log.debug(
            "world.stub_mode",
            neo4j_uri=self.neo4j_uri,
            note="real Graphiti integration deferred",
        )

    # ---------- API (stubs) ----------

    def add_episode(
        self,
        name: str,
        episode_body: str,
        reference_time: Any | None = None,
        source_description: str = "",
    ) -> None:
        """В v0 — no-op. Реальный Graphiti: client.add_episode(...)."""
        log.debug(
            "world.add_episode.stub", name=name, length=len(episode_body)
        )

    def search(
        self,
        query: str,
        k: int = 10,
        valid_at: Any | None = None,
    ) -> list[dict[str, Any]]:
        """В v0 — пустой результат."""
        log.debug("world.search.stub", query=query, k=k)
        return []
