# harnes

Research wrapper around local LLMs implementing a meta-cycle over a ReAct loop.

Цель — система, которая по целям из не заданного на этапе проектирования
распределения доводит их до завершения в консеквенциальном пространстве
действий на протяжённом горизонте при минимальном пошаговом контроле оператора.

Полная архитектура и журнал решений — `agent_architecture.html`.

## Статус

v0 — вертикальный срез в работе.

## Quick start

Поднять memory-бэкенды:

```bash
docker compose up -d qdrant neo4j
```

Собрать и запустить агент:

```bash
docker compose build agent
docker compose run --rm agent
```

Или непрерывно:

```bash
docker compose up agent
```

## Локальная разработка (без docker)

```bash
uv sync
uv run python scripts/run_agent.py
uv run pytest
```

## Конфигурация

Дефолты — `config/default.yaml`. Переопределение через env-переменные
(см. `docker-compose.yaml` и `src/harnes/config.py`).

## LLM endpoint

В v0 — внешний `ll-router` на `http://192.168.0.111:8000/v1`,
модель `gemma-26b-a4b`. Подключение через LiteLLM.

## Admin-консоль (webui)

Отдельный сервис в том же compose. Read-friendly UI для наблюдения за
агентом + минимальное управление целями (approve/reject/create). См.
`webui/README.md`.

```bash
docker compose up -d webui
open http://127.0.0.1:8080
```
