"""Refuse shutdown self-wait from callbacks and their inherited child context."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from kohakuterrarium.errors import ConflictError

_CALLBACK = ContextVar("kt_shutdown_callback", default=None)


@contextmanager
def shutdown_callback_scope(agent):
    """Mark callback descendants as wait-ineligible until their root task finishes."""
    token = _CALLBACK.set((agent, asyncio.current_task()))
    try:
        yield
    finally:
        _CALLBACK.reset(token)


def stop_wait_ineligible(operation):
    """Reject blocking stop within an input or turn callback's live task lineage."""

    @wraps(operation)
    async def scoped(agent, *args, **kwargs):
        with shutdown_callback_scope(agent):
            return await operation(agent, *args, **kwargs)

    return scoped


def reject_shutdown_callback_wait(agent):
    """Deny an inherited callback self-wait without granting lifecycle authority."""
    scope = _CALLBACK.get()
    if scope is not None and scope[0] is agent and not scope[1].done():
        raise ConflictError("Cannot stop and wait from an active agent callback")
