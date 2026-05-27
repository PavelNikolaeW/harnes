"""Tick journal — list / stats / SSE live feed."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from harnes.metacycle.journal import TickEventType, TickJournal
from harnes.webui.deps import get_journal
from harnes.webui.templating import templates

router = APIRouter()


@router.get("", response_class=HTMLResponse)
def list_events(
    request: Request,
    event_type: str | None = None,
    tick_id: int | None = None,
    limit: int = 100,
    journal: TickJournal = Depends(get_journal),
) -> HTMLResponse:
    """Recent N событий + stats + последний snapshot."""
    limit = max(1, min(limit, 1000))
    et: TickEventType | None = None
    if event_type:
        try:
            et = TickEventType(event_type)
        except ValueError:
            raise HTTPException(400, f"unknown event_type: {event_type}")

    events = journal.recent_events(limit=limit, event_type=et, tick_id=tick_id)
    stats = journal.stats()
    snap = journal.latest_snapshot()

    return templates.TemplateResponse(
        request,
        "journal/list.html",
        {
            "events": events,
            "stats": stats,
            "snapshot": snap,
            "selected_event_type": event_type,
            "selected_tick_id": tick_id,
            "limit": limit,
            "all_event_types": [e.value for e in TickEventType],
        },
    )


@router.get("/sse")
async def sse_journal(
    request: Request,
    journal: TickJournal = Depends(get_journal),
    poll_interval: float = 2.0,
):
    """SSE: tail tick journal. Каждые N сек — новые events с момента last_seen.

    Клиент: <div hx-ext="sse" sse-connect="/journal/sse" sse-swap="event">…
    На сервере event-name = "event" (htmx-sse удобство).
    """

    async def event_stream():
        last_seen: datetime | None = datetime.now(UTC)
        while True:
            if await request.is_disconnected():
                break

            since = last_seen
            new = journal.recent_events(limit=50, since=since)
            new = list(reversed(new))  # старые → новые

            for ev in new:
                payload = {
                    "id": ev.id,
                    "ts": ev.timestamp.isoformat() if ev.timestamp else "",
                    "tick_id": ev.tick_id,
                    "event_type": ev.event_type,
                    "payload": ev.payload_json,
                }
                yield {
                    "event": "event",
                    "data": json.dumps(payload),
                }
                if ev.timestamp:
                    last_seen = ev.timestamp

            await asyncio.sleep(max(0.5, poll_interval))

    return EventSourceResponse(event_stream())
