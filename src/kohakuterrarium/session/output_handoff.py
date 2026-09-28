"""Resolve async output consumers after a source binding reservation is released."""

import asyncio
from concurrent.futures import Future
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


_ACTIVE = ContextVar("kt_accepted_output", default=None)


def _task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _accepted(sink):
    task = _task()
    active = _ACTIVE.get()
    return (active is not None and active == (task, sink)) or (
        task is not None and getattr(task, "_topology_deferred", None) is False
    )


def defer_output(operation):
    """Defer a synchronous output event before any store-bound writer is captured."""

    @wraps(operation)
    def admitted(sink, *args, **kwargs):
        agent = getattr(sink, "_agent", None)
        lease = getattr(agent, "_topology_handoff_task", None)
        if lease is None or lease is _task() or _accepted(sink):
            return operation(rebound_output(sink), *args, **kwargs)
        args, kwargs = deepcopy((args, kwargs))
        receipt = Future()
        pending = getattr(agent, "_deferred_output_calls", None)
        if pending is None:
            pending = agent._deferred_output_calls = set()

        async def publish():
            try:
                target = await output_after_handoff(sink)
                result = operation(target, *args, **kwargs)
                if isinstance(result, Future):
                    result = await asyncio.shield(asyncio.wrap_future(result))
                if not receipt.done():
                    receipt.set_result(result)
            except BaseException as exc:
                logger.error("Deferred session output failed", error=str(exc))
                if not receipt.done():
                    receipt.set_exception(exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise

        task = asyncio.create_task(publish())
        pending.add(task)
        task.add_done_callback(pending.discard)
        return receipt

    return admitted


@defer_output
def persist_scratchpad(sink, values):
    """Persist a detached pad through the admitted sink binding."""
    return sink._submit_store_write(
        sink._store.save_state, sink._event_key_prefix, scratchpad=deepcopy(values)
    )


async def drain_deferred_output(sink):
    agent = getattr(sink, "_agent", None)
    if getattr(agent, "_topology_handoff_task", None) is _task():
        return
    pending = tuple(getattr(agent, "_deferred_output_calls", ()))
    if pending:
        await asyncio.gather(*(asyncio.shield(task) for task in pending))


def admit_output(operation):
    """Track one finite async output call independently of its controller task."""

    @wraps(operation)
    async def admitted(sink, *args, **kwargs):
        args, kwargs = deepcopy((args, kwargs))
        async with output_binding(sink) as target:
            return await operation(target, *args, **kwargs)

    return admitted


@asynccontextmanager
async def output_binding(sink):
    """Reserve a finite capture/write/install operation against topology handoff."""
    if sink is None or _accepted(sink):
        yield sink
        return
    sink = await output_after_handoff(sink)
    receipt = asyncio.get_running_loop().create_future()
    pending = getattr(sink, "_inflight_output_calls", None)
    if pending is None:
        pending = sink._inflight_output_calls = set()
    pending.add(receipt)
    token = _ACTIVE.set((asyncio.current_task(), sink))
    try:
        yield sink
    finally:
        _ACTIVE.reset(token)
        receipt.set_result(None)
        pending.discard(receipt)


def rebound_output(sink):
    agent = getattr(sink, "_agent", None)
    if agent is None:
        return sink
    if getattr(agent, "_session_attach_uncertain", False):
        raise ConflictError("Output binding requires cold handoff recovery")
    seen = set()
    while (current := getattr(sink, "_handoff_successor", None)) is not None:
        if id(sink) in seen or current._event_key_prefix != sink._event_key_prefix:
            raise ConflictError("Output handoff successor is invalid")
        seen.add(id(sink))
        sink = current
    return sink


async def output_after_handoff(sink):
    """Wait before buffering or capturing a store-bound operation, then select its sink."""
    agent = getattr(sink, "_agent", None)
    while getattr(agent, "_topology_handoff_task", None) is not None:
        if agent._topology_handoff_task is asyncio.current_task() or _accepted(sink):
            return sink
        await agent._topology_handoff_ready.wait()
    return rebound_output(sink)
