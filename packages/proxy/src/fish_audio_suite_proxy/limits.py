"""Request body size cap, enforced before any route reads the body."""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fish_audio_suite_proxy.errors import ProxyError, json_error

_TOO_LARGE = 413


class BodyTooLargeError(ProxyError):
    """Signal that the request body went past ``FISH_PROXY_MAX_BODY_BYTES``."""

    def __init__(self, limit: int) -> None:
        """Build the 413 for ``limit`` bytes.

        Parameters
        ----------
        limit : int
            The configured cap, shown to the client.
        """
        super().__init__(_TOO_LARGE, f"request body exceeds {limit} bytes")


def _limit(scope: Scope) -> int:
    app = scope.get("app")
    settings = getattr(getattr(app, "state", None), "settings", None)
    return int(getattr(settings, "max_body_bytes", 0) or 0)


class BodyLimitMiddleware:
    """Reject a body over the cap, by ``Content-Length`` or by counting bytes.

    Notes
    -----
    A chunked upload has no length, so the received bytes are counted and
    the read raises ``BodyTooLargeError``. A cap of 0 turns the check off. The
    cap lives on ``app.state.settings``, which the lifespan fills in.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``.

        Parameters
        ----------
        app : ASGIApp
            The next application in the stack.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check the body size, then pass the request on.

        Parameters
        ----------
        scope : Scope
            ASGI scope. Only ``http`` requests are checked.
        receive : Receive
            Body channel, wrapped to count bytes.
        send : Send
            Response channel.
        """
        limit = _limit(scope) if scope["type"] == "http" else 0
        if limit <= 0:
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            refusal = json_error(_TOO_LARGE, f"request body exceeds {limit} bytes")
            await refusal(scope, receive, send)
            return
        seen = 0

        async def counted() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise BodyTooLargeError(limit)
            return message

        await self.app(scope, counted, send)
