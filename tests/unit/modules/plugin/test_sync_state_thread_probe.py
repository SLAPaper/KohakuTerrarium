"""Test-only sync-hook bridge; production dispatch remains unchanged."""

import asyncio
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kohakuterrarium.modules.plugin.base import PluginContext
from kohakuterrarium.modules.plugin.state_io import (
    read_plugin_state,
    write_plugin_state,
)
from kohakuterrarium.modules.plugin.state_scope import context_state_key
from kohakuterrarium.session.store import SessionStore


class _ThreadStateContext:
    def __init__(self, context):
        self._store = context.session_store
        self._prefix = context_state_key(context, "")
        self._loop_thread = threading.get_ident()

    def get_state(self, key):
        assert threading.get_ident() != self._loop_thread
        return self._store.submit(
            read_plugin_state, self._store, self._prefix + key
        ).result()

    def set_state(self, key, value):
        assert threading.get_ident() != self._loop_thread
        self._store.submit(
            write_plugin_state, self._store, self._prefix + key, deepcopy(value)
        ).result()


async def _invoke_sync_hook(hook, context):
    detached = _ThreadStateContext(context)
    task = asyncio.create_task(asyncio.to_thread(hook, detached))
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            break
    try:
        result = task.result()
    except BaseException as exc:
        if cancelled is not None:
            raise cancelled from exc
        raise
    if cancelled is not None:
        raise cancelled
    return result


def _context(store):
    host = SimpleNamespace(config=SimpleNamespace(name="alice"), session_store=store)
    return PluginContext(agent_name="display", _host_agent=host, _plugin_name="old")


@pytest.mark.parametrize("failure", [False, True])
async def test_sync_hook_waits_off_loop_for_affinity_ack(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "thread.kohakutr"
    store = SessionStore(path)
    context = _context(store)
    prefix = context_state_key(context, "count")
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    access_threads = []
    hook_threads = []
    put = type(store.state).__setitem__

    def observed(table, key, value):
        if table is store.state and key == prefix:
            access_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), "state barrier not released"
            if failure:
                raise OSError("state commit failed")
        return put(table, key, value)

    def legacy_hook(ctx):
        hook_threads.append(threading.get_ident())
        ctx.set_state("count", (ctx.get_state("count") or 0) + 1)
        return ctx.get_state("count")

    monkeypatch.setattr(type(store.state), "__setitem__", observed)
    task = asyncio.create_task(_invoke_sync_hook(legacy_hook, context))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        heartbeat = asyncio.get_running_loop().create_future()
        loop.call_soon(heartbeat.set_result, True)
        assert await heartbeat
        assert not task.done()
        assert hook_threads[0] != threading.get_ident()
        assert access_threads == [store._affinity_worker_ident]
        assert access_threads[0] != hook_threads[0]
        release.set()
        if failure:
            with pytest.raises(OSError, match="state commit failed"):
                await task
        else:
            assert await task == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(store.close)
    reopened = SessionStore(path)
    try:
        assert reopened.state.get(prefix) == (None if failure else 1)
    finally:
        reopened.close()


async def test_cancelled_hook_is_joined_before_store_close(tmp_path):
    store = SessionStore(tmp_path / "cancel.kohakutr")
    context = _context(store)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    def legacy_hook(ctx):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "hook barrier not released"
        ctx.set_state("terminal", "finished")

    task = asyncio.create_task(_invoke_sync_hook(legacy_hook, context))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await context.get_state_async("terminal") == "finished"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(store.close)
