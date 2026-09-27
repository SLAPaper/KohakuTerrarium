"""Task-local authorization for one hosted provider request's recovery."""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from kohakuterrarium.errors import ConflictError


@dataclass
class RecoveryScope:
    """Request identity and host callbacks active only during provider execution."""

    provider: Any
    task: Any
    messages: list[dict]
    commit: Any
    rescue: Any


_ACTIVE: ContextVar[RecoveryScope | None] = ContextVar(
    "kt_recovery_scope", default=None
)


def active_recovery(provider=None, messages=None):
    scope = _ACTIVE.get()
    if scope is None or scope.task is not asyncio.current_task():
        return None
    if provider is not None and scope.provider is not provider:
        return None
    if messages is not None and scope.messages is not messages:
        raise ConflictError("Recovery request changed outside its authorized retry")
    return scope


async def scoped_chat(provider, messages, scope, **kwargs):
    """Advance and close a provider iterator without leaking authority to consumers."""
    iterator = provider.chat(messages, **kwargs).__aiter__()
    try:
        while True:
            token = _ACTIVE.set(scope)
            try:
                chunk = await iterator.__anext__()
            except StopAsyncIteration:
                return
            finally:
                _ACTIVE.reset(token)
            yield chunk
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            token = _ACTIVE.set(scope)
            try:
                await close()
            finally:
                _ACTIVE.reset(token)
