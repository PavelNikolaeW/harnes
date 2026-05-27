"""HTTP Basic Auth middleware для webui — optional.

Включается если в env заданы оба `WEBUI_AUTH_USERNAME` + `WEBUI_AUTH_PASSWORD`.
Иначе middleware не подключается (compose-default = loopback-only без auth).

Безопасность:
- `secrets.compare_digest` против timing-attack.
- Realm "harnes-webui" — клиенту видно префикс.
- 401 + WWW-Authenticate Header — браузер показывает native login prompt.
- Static и /health тоже под auth (research-console целиком приватная).
"""
from __future__ import annotations

import base64
from secrets import compare_digest

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_REALM = "harnes-webui"
_UNAUTHORIZED_BODY = (
    "<!doctype html><html><body><pre>401 Unauthorized · harnes-webui</pre></body></html>"
)


def _unauthorized() -> Response:
    return Response(
        content=_UNAUTHORIZED_BODY,
        status_code=401,
        media_type="text/html",
        headers={"WWW-Authenticate": f'Basic realm="{_REALM}"'},
    )


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Простой Basic-Auth. Подключается в create_app() если username непуст."""

    def __init__(self, app, username: str, password: str) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        # bytes для compare_digest — устраняет ambiguity str vs bytes.
        self._user_bytes = username.encode("utf-8")
        self._pwd_bytes = password.encode("utf-8")

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("basic "):
            return _unauthorized()
        try:
            decoded = base64.b64decode(auth[6:].strip()).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return _unauthorized()
        user, sep, pwd = decoded.partition(":")
        if not sep:
            return _unauthorized()
        user_ok = compare_digest(user.encode("utf-8"), self._user_bytes)
        pwd_ok = compare_digest(pwd.encode("utf-8"), self._pwd_bytes)
        if not (user_ok and pwd_ok):
            return _unauthorized()
        return await call_next(request)
