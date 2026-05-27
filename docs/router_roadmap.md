# LL-router doplniate — roadmap для агента Harnes

> **Цель документа.** Зафиксировать минимальный набор API-расширений ll-router
> (`http://192.168.0.111:8000`), которые принесут наибольшую отдачу для
> агента `harnes`. Приоритизированы по соотношению ценности к стоимости
> реализации.

## Что сейчас экспонирует роутер

```
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions          (legacy)
```

Это покрывает только LLM-инференс. Всё остальное (embeddings, rerank, health,
stats) либо отсутствует, либо реализовано локально на dev-машине агента.

## Состояние агента на 2026-05-27

- v1.0 закрыта (#31-#35), 328 тестов проходят.
- В коде уже есть **feature-flag `embeddings.use_server`** в
  `src/harnes/config.py:EmbeddingsConfig` — переключение между серверным
  embedding endpoint'ом и локальным fastembed-фолбэком.
- Сейчас `use_server=false` по умолчанию (раз endpoint'а нет).
- Также есть `_embed_via_server` через LiteLLM — будет работать сразу, как
  только endpoint появится.

---

## Tier 1 — высокая отдача

### Task R1 — `POST /v1/embeddings` ★★★ (must-have)

**Use cases в коде агента:**

| место | что embed'ит | размер batch |
|-------|--------------|--------------|
| `memory/semantic.py` (Qdrant write) | факты при сохранении | 1 за раз |
| `memory/router._recall_semantic` | query при recall | 1 за раз |
| `eval/multi_turn.InMemoryChunkStore.add_chunks` | chunks длинного MAB-context | 20-50 batch |
| `eval/multi_turn.search` | query при tool=recall_memory | 1 за раз |
| `tools/builtin/recall.recall_impl` | через MemoryRouter, см. выше | 1 за раз |

**Что сейчас делает агент без endpoint'а:**

- `src/harnes/llm/embeddings.py` использует `fastembed` локально.
- Дефолтная модель: `paraphrase-multilingual-mpnet-base-v2` (~500MB ONNX).
- Latency на CPU dev-машины: ~200-500ms на один текст, 5-15s на batch из 20-50 chunks.
- При первом запуске агента модель скачивается (~500MB) и хранится в `~/.cache/`.

**Ожидаемый профит при endpoint'e на GPU роутера:**

- BGE-M3 на GPU: ~10ms на один embed, ~50-100ms на batch из 50.
- Ускорение MAB multi-turn task setup: 5-15s → 0.5s (× 10-30).
- Никакой ONNX-зависимости на dev-машине.
- Shared cache между всеми клиентами роутера.

**Контракт (OpenAI-compatible — будет drop-in совместим с LiteLLM):**

```http
POST /v1/embeddings
Content-Type: application/json

{
  "model": "bge-m3",
  "input": ["text1", "text2"]
}
```

Response:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0},
    {"object": "embedding", "embedding": [0.3, 0.4, ...], "index": 1}
  ],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 8, "total_tokens": 8}
}
```

**Рекомендуемая модель:** **BGE-M3** (multilingual, 1024-dim, 8k context).
Альтернативы по падающему качеству / растущей скорости:
- `BAAI/bge-large-en-v1.5` (en-only, 1024d)
- `BAAI/bge-small-en-v1.5` (en-only, 384d)
- `paraphrase-multilingual-mpnet-base-v2` (multilingual, 768d, sentence-transformers)

**Performance budget:**
- p50 latency на одиночный текст ≤ 30ms
- batch=50 ≤ 200ms
- throughput ≥ 100 req/s

**Что включить на стороне агента после deploy:**

```yaml
# config/default.yaml
embeddings:
  use_server: true
  model: "bge-m3"
```

Граmceful fallback на fastembed уже реализован — при 404/timeout агент
автоматически свалится на локальный embed.

---

### Task R2 — `GET /health` ★★ (very nice-to-have)

**Use case:** агент в долгом `run-loop --real` хочет дешёвый precheck перед
каждым LLM-вызовом или хотя бы при старте. Сейчас падение роутера ловится
через LiteLLM-timeout (`settings.llm.timeout = 60s`) — это **60 секунд впустую
на каждый зависший action_call**.

Похожая проблема была с Neo4j: было 15s timeout у драйвера, заменили на
2-секундный TCP precheck (`memory/world.py:_bolt_reachable`). Хочется такой же
паттерн для роутера, но HTTP-based.

**Контракт:**

```http
GET /health
```

Response (любой из вариантов):

```json
{"status": "ok", "uptime_seconds": 12345, "version": "b391-2e97c5f96"}
```

Или просто:

```http
HTTP/1.1 200 OK
```

**Performance budget:** p50 ≤ 10ms, всегда отвечает (даже при перегруженном
inference-pool'е).

**Что агент будет делать:**

- В `llm/client.py` появится helper `is_router_healthy(timeout_s=2) -> bool`.
- В `run-loop --real` старт-checklist:
  - если роутер не healthy → fallback в `--stub` mode с warning.
  - либо ждать `--router-wait-seconds N` с экспоненциальным backoff.

---

### Task R3 — `GET /v1/stats` или `/v1/models/load_status` ★★

**Use case:** сегодня (2026-05-26) обнаружили, что `gemma-26b-a4b` молча
fallback'нулась на CPU-вариант, и latency упала с 146 t/s до ~12 t/s. Если бы
агент мог при старте узнать, что модель сейчас на CPU — мог бы:

1. Логнуть warning (operator увидит).
2. Уменьшить `max_steps` / `repeat_k` чтобы не тратить часы.
3. Перейти на меньшую модель (если есть в tier'ах).

**Контракт (любой формат, главное стабильный):**

```http
GET /v1/models/load_status
```

```json
{
  "models": [
    {"id": "gemma-26b-a4b", "backend": "gpu", "device": "cuda:0", "loaded": true, "queue_depth": 0},
    {"id": "qwen-35b", "backend": "cpu", "device": "cpu", "loaded": false, "queue_depth": 0}
  ]
}
```

Или просто расширение `/v1/models`:

```json
{
  "data": [
    {"id": "gemma-26b-a4b", "backend": "gpu", "throughput_tps": 146, ...}
  ]
}
```

**Performance budget:** p50 ≤ 20ms.

---

## Tier 2 — средняя отдача

### Task R4 — `POST /v1/rerank` ★

**Use case:** Graphiti (наш world model в `memory/world.py`) требует
cross-encoder reranker. Сейчас стоит `NoopReranker` — возвращает score=1.0 для
всех результатов KG-поиска. Это значит, что порядок зависит только от
embedding similarity, без переранжирования по релевантности query.

Реальный reranker улучшил бы качество retrieval, особенно на:
- Long-range MAB задачах (нужно ранжировать какой chunk из 50 наиболее релевантен)
- Conflict Resolution (нужно отличить актуальный факт от устаревшего)

**Контракт (Cohere/Voyage-style):**

```http
POST /v1/rerank
{
  "model": "bge-reranker-v2-m3",
  "query": "what did the agent compute",
  "documents": ["doc1 text", "doc2 text", "doc3 text"],
  "top_k": 3
}
```

Response:

```json
{
  "results": [
    {"index": 1, "relevance_score": 0.92, "document": "doc2 text"},
    {"index": 0, "relevance_score": 0.45, "document": "doc1 text"},
    {"index": 2, "relevance_score": 0.10, "document": "doc3 text"}
  ],
  "model": "bge-reranker-v2-m3"
}
```

**Рекомендуемая модель:** `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder).

**Performance budget:** ≤ 100ms для query + 10 documents.

**Включение на стороне агента:** заменить `NoopReranker` в
`memory/world.py:_init_graphiti` на rerank-клиент с тем же fallback'ом
(`use_server` → server, иначе noop).

---

### Task R5 — Streaming `POST /v1/chat/completions` с `stream: true` ★

Скорее всего **уже работает** на стороне роутера (это стандарт OpenAI), но я не
проверял. Если работает — пометить как готовое. Если нет — реализовать.

**Use case:**
- `react/loop.py:_action_call` мог бы early-stop при детекции `"tool_id": "finish"` (экономия 50-200 токенов на trajectory)
- "Thinking..." UX в CLI `run-loop`

**Бенефит:** маржинальный, можно отложить.

---

## Tier 3 — низкая отдача (опционально / на будущее)

### R6. `POST /v1/batch` — батч-инференс для eval

Сейчас `eval/harness.run_evaluation` запускает задачи **последовательно**.
Прогон MAB с `--hf-examples-per-split 10 --repeat-k 3` = ~120 запросов × ~30s/запрос
= **час**.

С batch endpoint:
- Submit 120 задач → async-ack → poll for completion
- Реалистично сжать до 5-10 минут (зависит от model parallelism)

Не блокер для research-валидации, но улучшает throughput.

### R7. Native tool-use API (OpenAI `tools: [...]` spec)

Gemma не tool-tuned, поэтому сейчас мы парсим JSON в `_parse_action_json` через
regex. Работает на 95% случаев. Если роутер добавит structured tool-call (через
constrained decoding или GBNF) — упростит `react/loop.py`. Но не критично.

### R8. `/v1/files` для длинных контекстов

Отпало после v0.3 #27 multi-turn chunk-injection — мы научились резать
длинный context на 2k-char chunks и инжектить через `recall_memory` tool. Так
что `/v1/files` нам уже не нужен.

---

## Сводная таблица

| ID | Endpoint | Priority | Profit | Реализовано на стороне агента |
|----|----------|----------|--------|------------------------------|
| R1 | `POST /v1/embeddings` | ★★★ | ×10-30 на multi-turn setup | ✓ feature-flag `use_server` |
| R2 | `GET /health` | ★★ | -60s timeout на failure | ✗ (нужен helper) |
| R3 | `GET /v1/models/load_status` | ★★ | detect CPU fallback | ✗ (опционально) |
| R4 | `POST /v1/rerank` | ★ | +retrieval quality на long-range | ✗ (нужен клиент в Graphiti) |
| R5 | `stream: true` | ★ | -50-200 токенов/trajectory | ✓ через LiteLLM |
| R6 | `POST /v1/batch` | ◇ | ×10 на eval throughput | ✗ |
| R7 | tool-use spec | ◇ | code simplification | ✗ |
| R8 | `/v1/files` | — | n/a (отпало) | — |

## TL;DR что просить в первую очередь

1. **R1: `/v1/embeddings` с BGE-M3** — самый ощутимый профит. Уберёт 500MB
   ONNX-модель с dev-машины, ускорит MAB multi-turn setup в 10-30×. Агент
   готов к нему — достаточно выставить `embeddings.use_server: true`.

2. **R2: `/health`** — тривиально реализовать, но сильно улучшает UX
   при flaky-сети или перегрузке.

3. **R3: load_status** — поможет ловить сегодняшнюю ситуацию (CPU fallback)
   до того, как кто-то напишет «странно медленно».

Остальное по мере появления свободного времени или конкретного use case.
