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

## Что есть

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
| World KG graph    | `/memory/world/graph` | Cytoscape-визуализация temporal KG; data via `/memory/world/cytoscape.json` |
| Skills            | `/skills`          | Список бандлов с агрегированными метриками; deprecated не теряются          |
| Skill detail      | `/skills/{id}`     | Prompt template, allowed tools, per-version history метрик                  |
| Eval history      | `/eval`            | Прогоны benchmark'а с фильтрами; held-out скрыты по умолчанию               |
| Eval detail       | `/eval/{id}`       | Полные метрики прогона + failure modes + snapshot skill_versions            |
| Eval compare      | `/eval/compare?base=&cand=` | Side-by-side diff двух прогонов (как CLI eval-compare)            |
| Commands          | `/commands`        | Web→agent IPC: pause / resume / trigger_tick + история, status loop'а       |
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
- Запуск/остановка контейнера — за docker compose / CLI; webui не управляет
  жизненным циклом процесса.
- Pause / resume / trigger_tick — через `commands` channel: webui пишет
  команду в `web_commands.db`, agent `run-loop` drain'ит её в начале каждой
  итерации перед `sense`. См. `harnes/metacycle/commands.py`.
- Risk: два процесса (agent run-loop + webui POST) пишут в SQLite файлы.
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
- HTTP Basic Auth — опциональный. Включается env-vars:
  ```bash
  WEBUI_AUTH_USERNAME=admin WEBUI_AUTH_PASSWORD=$(openssl rand -hex 16)
  ```
  Если username пуст — auth выключен (default). Использовать когда выйдешь
  за loopback (общий dev, reverse proxy без TLS — недопустимо).
- Реверс-прокси (Caddy / nginx) с TLS — обязателен для production exposure.

## Tailwind: CDN vs pre-built

- **Dev (default)** — Tailwind через CDN (`cdn.tailwindcss.com`, JIT в браузере,
  ~150KB JS). Удобно для итераций — никаких build-steps.
- **Production** — pre-built `static/css/tailwind.css`, ~20KB после purge.
  Собирается в Dockerfile через standalone CLI (`webui/tailwind/build.sh`),
  без node_modules. `base.html` авто-переключается на pre-built если файл есть.

Локально пересобрать (например после нового template):
```bash
./webui/tailwind/build.sh           # production minified
./webui/tailwind/build.sh --watch   # dev watch mode
```

## Limitations (известные, follow-ups)

- **Запуск/остановка процесса.** Контейнер по-прежнему стартует через
  `docker compose`; webui не убивает и не запускает процессы. Pause/resume
  и trigger_tick — только для уже работающего `run-loop`.
- **Semantic search требует embeddings.** Если `fastembed`-fallback недоступен
  и роутер не возвращает `/v1/embeddings` — поиск падает. UI показывает ошибку.
- **KG-визуализация.** Direct Cypher через neo4j-driver, LIMIT 200/200
  по дефолту. Большие графы (>1k) не оптимизированы.
- **Skill edit.** Read-only — изменения скиллов остаются за reflect и CLI.
- **UTF-8 в curl.** Если посылаешь POST на `/goals` через `curl --data` с
  кириллицей — без `--data-urlencode` будут битые байты. Браузерная форма
  работает корректно (стандарт `application/x-www-form-urlencoded` + UTF-8).

## Развитие

Естественные следующие шаги:

1. **Trajectory replay.** Архитектура (§16) обещает воспроизводимость —
   замечательно показать в UI rewind/forward по шагам с подсветкой того,
   что видел агент в каждый момент.
2. **Diff trajectories.** Side-by-side сравнение двух trajectory'ев одной цели
   (полезно после reflect-bump'а скилла).
3. **World KG filters.** Фильтр по labels / Cypher-snippet input на странице
   `/memory/world/graph`.
4. **Bulk goal actions.** Approve/reject/abandon нескольких целей за раз.
5. **Pause persistence.** Сейчас `paused`-flag живёт in-memory у run-loop;
   рестарт контейнера сбрасывает. Может стоит хранить в `web_commands.db`
   или TickJournal как latest state.
