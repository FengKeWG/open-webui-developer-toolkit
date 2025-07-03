"""
title: Input Inspector
id: input_inspector
author: OpenAI Codex
funding_url: https://github.com/jrkropp/open-webui-developer-toolkit
description:
    Emit citation blocks containing the contents of the input arguments passed into a pipe.
version: 0.1.1
license: MIT
notes:
    - Sensitive headers (such as 'authorization', 'cookie', and similar) are redacted by default
      to prevent leaking security-sensitive information into debugging logs. You can disable
      this by setting the `REDACT_REQUEST` valve to `False`.
    - See https://github.com/jrkropp/open-webui-developer-toolkit/blob/development/docs/pipe_input.md for more details.
"""

from __future__ import annotations

import datetime
import json
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional
from pydantic import BaseModel, Field
from fastapi import Request

SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "x-forwarded-for",
    "x-envoy-external-address",
}


class Pipe:
    class Valves(BaseModel):
        """Configurable pipe settings."""

        REDACT_REQUEST: bool = Field(
            default=True,
            description="Redact sensitive request headers in the citation output.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None,
        __files__: list[dict[str, Any]] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __tools__: dict[str, Any] | None = None,
        __task__: Optional[dict[str, Any]] = None,
        __task_body__: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None] | str:
        """Send citation blocks for each argument and return a short message."""

        async def emit(name: str, value: Any) -> None:
            if __event_emitter__ is None:
                return

            serial = _safe_json(value)
            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "document": [json.dumps(serial, indent=2)],
                        "metadata": [
                            {
                                "date_accessed": datetime.datetime.utcnow().isoformat(),
                                "source": name,
                            }
                        ],
                        "source": {"name": name},
                    },
                }
            )

        await emit("body", body)
        await emit("__metadata__", __metadata__ or {})
        await emit("__user__", __user__)
        await emit("__request__", _sanitize_request(__request__, self.valves.REDACT_REQUEST))
        await emit("__files__", __files__ or [])
        await emit("__tools__", __tools__ or {})
        if __task__:
            await emit("__task__", __task_body__)

        return "Input inspection complete. See citations for details."

def _sanitize_request(request: Request, redact: bool) -> dict[str, Any]:
    """Return a sanitized representation of ``request``."""

    headers = {
        k: ("[REDACTED]" if redact and k.lower() in SENSITIVE_HEADERS else v)
        for k, v in request.headers.items()
    }
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": headers,
    }

def _safe_json(obj: Any) -> Any:
    """Recursively convert obj to JSON-serializable form."""

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if hasattr(obj, "dict"):
        try:
            return _safe_json(obj.dict())
        except Exception:  # pragma: no cover - best effort
            pass
    if hasattr(obj, "model_dump"):
        try:
            return _safe_json(obj.model_dump())
        except Exception:  # pragma: no cover - best effort
            pass
    return f"<UNSERIALIZABLE {type(obj).__name__}>"
