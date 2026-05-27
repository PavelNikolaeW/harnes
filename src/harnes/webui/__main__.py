"""Dev-server entry point: `python -m harnes.webui`.

В production используется uvicorn напрямую (см. webui/Dockerfile CMD).
"""
from __future__ import annotations

import uvicorn

from harnes.webui.config import get_webui_settings


def main() -> None:
    cfg = get_webui_settings()
    uvicorn.run(
        "harnes.webui.app:create_app",
        factory=True,
        host=cfg.host,
        port=cfg.port,
        reload=cfg.reload,
        log_level=cfg.log_level.lower(),
    )


if __name__ == "__main__":
    main()
