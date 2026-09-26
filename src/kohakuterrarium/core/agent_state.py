"""Per-agent session-state owner lifecycle and acknowledged short commits."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from kohakuterrarium.core.session_state_owner import SessionStateOwner
from kohakuterrarium.errors import ConflictError, InvalidRequestError
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_session_state_owner(agent: Any) -> None:
    """Create or reuse the per-agent owner and start its control task."""
    owner = getattr(agent, "_session_state_owner", None)
    task = getattr(owner, "task", None)
    if owner is None or task is None or task.done():
        name = getattr(getattr(agent, "config", None), "name", "agent")
        owner = SessionStateOwner(name=f"session_state_owner_{name}")
        agent._session_state_owner = owner
    owner.start()
    agent.controller._session_state_owner = owner


async def rollback_failed_start(agent: Any) -> None:
    """Stop producers, the consumer, and the owner after an incomplete start."""
    agent._running = False
    shutdown = getattr(agent, "_shutdown_event", None)
    if shutdown is not None:
        shutdown.set()
    owner = getattr(agent, "_session_state_owner", None)
    close_admission = getattr(owner, "close_admission", None)
    if callable(close_admission):
        close_admission()
    consumer = getattr(agent, "_consumer_task", None)
    if consumer is not None:
        if not consumer.done():
            consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
    cleanups = []
    trigger_manager = getattr(agent, "trigger_manager", None)
    if trigger_manager is not None:
        cleanups.append(trigger_manager.stop_all)
    input_module = getattr(agent, "input", None)
    if input_module is not None:
        cleanups.append(input_module.stop)
    mcp = getattr(agent, "_mcp_manager", None)
    if mcp is not None:
        cleanups.append(mcp.shutdown)
    plugins = getattr(agent, "plugins", None)
    if plugins is not None:
        cleanups.append(plugins.unload_all)
    for cleanup in cleanups:
        try:
            await cleanup()
        except Exception:
            logger.warning("Failed to clean up incomplete startup", exc_info=True)
    if owner is not None:
        close = getattr(owner, "close", None)
        if callable(close):
            await close()
    output_router = getattr(agent, "output_router", None)
    if output_router is not None:
        try:
            await output_router.stop()
        except Exception:
            logger.warning("Failed to stop startup output", exc_info=True)


async def start_agent(agent: Any) -> None:
    """Start the owner, then existing modules, rolling back on failure."""
    stop_lock = getattr(agent, "_stop_lock", None)
    if stop_lock is not None and stop_lock.locked():
        raise ConflictError("Cannot start while the agent is stopping")
    if getattr(agent, "_starting", False):
        raise ConflictError("Agent startup is already in progress")
    if getattr(agent, "_running", False):
        owner = getattr(agent, "_session_state_owner", None)
        if owner is None or not owner.is_running:
            raise ConflictError("Running agent has no available session-state owner")
        return
    ensure_session_state_owner(agent)
    agent._starting = True
    try:
        await agent._start_modules()
    except BaseException:
        await rollback_failed_start(agent)
        raise
    finally:
        agent._starting = False


def run_session_state(agent: Any, callback: Callable[..., Any]) -> Any:
    """Submit a short sync commit, or run it inline on the owner task.

    Bootstrap before the owner has started may run inline on the caller.
    ``owner.submit`` itself still rejects re-entry; this helper must not
    deadlock by queueing work onto the task that is already running it.
    """
    if getattr(agent, "_stopped", False) or getattr(
        agent, "_model_switch_closing", False
    ):
        raise InvalidRequestError("session-state public admission is closed")
    owner = getattr(agent, "_session_state_owner", None)
    if owner is None or asyncio.current_task() is getattr(owner, "task", None):
        if inspect.iscoroutinefunction(callback) or inspect.isasyncgenfunction(
            callback
        ):
            raise InvalidRequestError(
                "session-state owner callback must be synchronous"
            )
        result = callback()
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise InvalidRequestError(
                "session-state owner callback must not return an awaitable"
            )
        return result
    return owner.submit(callback)


async def submit_session_state(agent: Any, callback: Callable[..., Any]) -> Any:
    """Await an accepted owner callback, preserving its error across cancel.

    A waiter cancelled after acceptance joins the callback, then re-raises
    ``CancelledError``. A callback error is attached as ``__cause__``.
    """
    result = run_session_state(agent, callback)
    if not asyncio.isfuture(result):
        return result
    cancelled = False
    callback_error: BaseException | None = None
    while True:
        try:
            value = await asyncio.shield(result)
        except asyncio.CancelledError as exc:
            if result.cancelled():
                if callback_error is not None and exc.__cause__ is None:
                    exc.__cause__ = callback_error
                raise
            cancelled = True
            continue
        except BaseException as exc:
            callback_error = exc
            if result.done():
                if cancelled:
                    raise asyncio.CancelledError() from exc
                raise
            continue
        else:
            if cancelled:
                raise asyncio.CancelledError()
            return value


async def submit_internal_session_state(
    agent: Any, callback: Callable[..., Any]
) -> Any:
    """Join an accepted internal finalization commit even while public admission closes."""
    owner = getattr(agent, "_session_state_owner", None)
    if owner is None or asyncio.current_task() is getattr(owner, "task", None):
        return callback()
    result = owner.submit_internal(callback)
    while True:
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            if result.cancelled():
                raise


def close_owner_admission(agent: Any) -> None:
    """Reject later public submits before stop awaits anything."""
    owner = getattr(agent, "_session_state_owner", None)
    close_admission = getattr(owner, "close_admission", None)
    if callable(close_admission):
        close_admission()


async def close_session_state_owner(agent: Any) -> None:
    """Drain accepted owner callbacks. Model-switch tasks are not on this owner."""
    owner = getattr(agent, "_session_state_owner", None)
    close = getattr(owner, "close", None)
    if callable(close):
        await close()
