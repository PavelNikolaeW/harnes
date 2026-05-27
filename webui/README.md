# harnes/webui — admin-консоль

Отдельный сервис: read-friendly консоль для исследовательского агента
**Irida** (см. `/agent_architecture.html`). Никакой autonomy — webui только
наблюдает состояние и принимает простые operator-команды по целям.

## Запуск

В составе compose-стека (рекомендуется):

```bash
docker compose up -d webui
open http://127.0.0.1:8080
```

Локально без docker (для разработки):

```bash
uv sync --extra webui

# Если данные агента лежат не в /app/data, переопределить пути:
export GOAL_STORE__SQLITE_PATH=./data/goals.db
export METACYCLE__JOURNAL_DB_PATH=./data/metacycle_journal.db
export MEMORY__LANCEDB_PATH=./data/lancedb
export EVAL__HISTORY_DB_PATH=./data/eval_history.db
export PROCEDURAL_STORE__SQLITE_PATH=./data/skill_metrics.db
export PROCEDURAL_STORE__BUNDLES_DIR=./skills

# Опционально:
export WEBUI_PORT=8080 WEBUI_RELOAD=true

uv run python -m harnes.webui
```

## Что есть в MVP

| View              | URL                | Что показывает                                                             |
|-------------------|--------------------|----------------------------------------------------------------------------|
| Dashboard         | `/dashboard`       | Сводка целей, recent tick events, recent trajectories                       |
| Goals             | `/goals`           | Список с фильтрами (status / class / origin), форма создания, approve/reject |
| Goal detail       | `/goals/{id}`      | Полный объект, predicate, budget, дерево потомков, depends_on               |
| Trajectories      | `/trajectories`    | Recent N трейекторий из LanceDB                                              |
| Trajectory inspector | `/trajectories/{id}` | Timeline шагов с typed-views для thought/plan/action/observation/critique |
| Tick journal      | `/journal`         | Events + stats + последний snapshot, фильтры по event_type/tick_id          |
| Tick live feed    | `/journal` (toggle) | SSE-стрим новых events каждые ~2s                                          |
| Memory explorer   | `/memory`          | Tabs: episodic (keyword), semantic (vector), world (Graphiti KG)            |
| Health            | `/health`          | TCP-чек Qdrant / Neo4j / LLM router + статус in-process stores              |
| OpenAPI           | `/docs`, `/openapi.json` | автогенерация FastAPI                                                  |

## Архитектурные решения

**Backend.** FastAPI + Jinja2 + HTMX + Alpine. Server-side rendering — минимум
JS, без отдельного front-build. SSE через `sse-starlette` для live feed.

**Связь с агентом.** Прямой read/write поверх тех же persistent stores, что
использует agent-контейнер (через bind-mount `./data`). Никакого HTTP API
внутри агента. Это значит:

- Approve/reject/create goal — webui пишет напрямую в `goals.db`; агент
  подхватит на следующем тике через `goal_arbitration`.
- Запуск/остановка `run-loop` — **остаётся за CLI**, webui это не делает.
  Тикать вручную из UI нет смысла, пока стоит автоматический loop.
- Risk: два процесса (agent run-loop + webui POST) пишут в один SQLite.
  В нашем dev-объёме это не проблема (SQLite WAL хорошо себя ведёт под
  такой нагрузкой), но для записи стоит держать webui-операции редкими
  (approve/reject/create — не каждую секунду).

**Tolerance.** Дашборд и health-страница не падают, если какой-то backend
недоступен (Qdrant down, Neo4j down). Каждая секция рендерит свою заглушку
("episodic store недоступен"). Это критично для исследовательских прогонов —
не должно быть так, что один битый neo4j рушит весь UI.

**Read-only mode.** Поставить `WEBUI_READ_ONLY=true` — спрячет кнопки
approve/reject/create. Удобно при demo / для второго слушателя в той же сессии.

## ENV-переменные

| Variable                      | Default                | Что делает                                |
|-------------------------------|------------------------|-------------------------------------------|
| `WEBUI_HOST`                  | `0.0.0.0`              | bind address                              |
| `WEBUI_PORT`                  | `8000`                 | внутренний порт контейнера (8080 на хосте)|
| `WEBUI_RELOAD`                | `false`                | auto-reload для dev                       |
| `WEBUI_LOG_LEVEL`             | `INFO`                 | logging level                             |
| `WEBUI_READ_ONLY`             | `false`                | спрятать write-actions                    |
| `GOAL_STORE__SQLITE_PATH` и пр. | из `harnes.config.Settings` | переопределение путей до stor'ов     |

## Безопасность

- В compose порт `8080` биндится на `127.0.0.1` — наружу контейнер не торчит.
- Auth нет — это single-user research-сетап. Если будет нужно выйти за loopback,
  добавь обратный прокси (Caddy / nginx) с basic-auth.

## Limitations (известные, follow-ups)

- **No write IPC с агентом.** Невозможно поставить агента на паузу или
  триггерить тик из UI. Запуск/остановка `run-loop` — через CLI.
- **Semantic search требует embeddings.** Если `fastembed`-fallback недоступен
  и роутер не возвращает `/v1/embeddings` — поиск падает. UI показывает ошибку.
- **World model (KG-визуализация).** Сейчас flat node-list; full graph view
  через Cytoscape — отложено.
- **Skills view.** Не в MVP. Реестр скиллов виден через CLI.
- **UTF-8 в curl.** Если посылаешь POST на `/goals` через `curl --data` с
  кириллицей — без `--data-urlencode` будут битые байты. Браузерная форма
  работает корректно (стандарт `application/x-www-form-urlencoded` + UTF-8).

## Развитие

Естественные следующие шаги:

1. **Command channel webui→agent.** Отдельная таблица `web_commands` в
   `goals.db`; agent run-loop её drain'ит на каждом тике. Это позволит
   pause/resume, trigger-tick, переопределить бюджеты.
2. **Skills view.** Список бандлов, history per-version метрик, diff между
   версиями. Read-only — изменения скиллов остаются через reflect.
3. **Eval-history view.** Тут уже есть store (`EvalHistoryStore`). Нужен
   только template + router.
4. **KG-визуализация** через Cytoscape.js. На уровне entity ≪10⁴ нод
   браузер хорошо тянет.
5. **Trajectory replay.** Архитектура (§16) обещает воспроизводимость —
   замечательно показать в UI rewind/forward по шагам с подсветкой того,
   что видел агент в каждый момент.
