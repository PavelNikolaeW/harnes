# harnes

Research wrapper around local LLMs implementing a meta-cycle over a ReAct loop.

Цель — система, которая по целям из не заданного на этапе проектирования
распределения доводит их до завершения в консеквенциальном пространстве
действий на протяжённом горизонте при минимальном пошаговом контроле оператора.

Полная архитектура и журнал решений — `agent_architecture.html`.

## Статус

v0 — вертикальный срез в работе.

## Quick start

```bash
cp .env.example .env       # отредактируй LLM_API_BASE если router не на 192.168.0.111
docker compose up -d       # qdrant + neo4j + agent (run-loop --real) + webui
open http://localhost:8080 # admin-консоль (Irida)
```

Логи агента:

```bash
docker compose logs -f agent
```

Остановить:

```bash
docker compose down        # сохраняет volumes (state)
docker compose down -v     # снести state (qdrant/neo4j data)
```

## Локальная разработка (без docker)

```bash
uv sync
uv run pytest

# Smoke агента
uv run python -m harnes.operator run-loop --stub --max-ticks 3

# UI на localhost:8000
WEBUI_PORT=8080 uv run python -m harnes.webui
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
