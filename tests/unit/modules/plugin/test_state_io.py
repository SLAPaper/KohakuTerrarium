"""Async plugin state uses the store worker and the creature-scoped key."""

import asyncio
import threading
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import PluginContext
from kohakuterrarium.modules.plugin.state_scope import context_state_key
from kohakuterrarium.modules.plugin.state_io import (
    read_plugin_state,
    write_plugin_state,
)
from kohakuterrarium.session.store import SessionStore


def _context(store, name, plugin="shared", display="stale-display-name"):
    host = SimpleNamespace(config=SimpleNamespace(name=name), session_store=store)
    return PluginContext(agent_name=display, _host_agent=host, _plugin_name=plugin)


async def _heartbeat(stop: asyncio.Event, stamps: list[float]) -> None:
    while not stop.is_set():
        stamps.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0)
    stamps.append(asyncio.get_running_loop().time())


async def test_async_state_keeps_loop_alive_on_store_worker(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "async.kohakutr")
    context = _context(store, "only")
    release = threading.Event()
    entered = threading.Event()
    real_get = type(store.state).get

    def slow_get(table, key, default=None):
        if table is store.state and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return real_get(table, key, default)

    stamps: list[float] = []
    stop = asyncio.Event()
    try:
        monkeypatch.setattr(type(store.state), "get", slow_get)
        beat = asyncio.create_task(_heartbeat(stop, stamps))
        await asyncio.sleep(0)
        task = asyncio.create_task(context.get_state_async("cursor"))
        assert await asyncio.to_thread(entered.wait, 2)
        assert len(stamps) >= 2
        release.set()
        assert await task is None
        await context.set_state_async("cursor", {"offset": 3})
        assert await context.get_state_async("cursor") == {"offset": 3}
        worker = await store.run(threading.get_ident)
        assert worker == store._affinity_worker_ident
        assert worker != threading.get_ident()
    finally:
        release.set()
        stop.set()
        await beat
        store.close()


async def test_worker_error_propagates(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "error.kohakutr")
    context = _context(store, "only")
    real_set = type(store.state).__setitem__

    def boom(table, key, value):
        if table is store.state and key == context_state_key(context, "cursor"):
            raise RuntimeError("state write failed")
        return real_set(table, key, value)

    monkeypatch.setattr(type(store.state), "__setitem__", boom)
    try:
        with pytest.raises(RuntimeError, match="state write failed"):
            await context.set_state_async("cursor", {"offset": 1})
    finally:
        store.close()


async def test_cancel_preserves_accepted_write(tmp_path, monkeypatch):
    path = tmp_path / "cancel.kohakutr"
    store = SessionStore(path)
    context = _context(store, "only")
    release = threading.Event()
    started = asyncio.Event()
    loop = asyncio.get_running_loop()
    real_set = type(store.state).__setitem__
    scoped = context_state_key(context, "cursor")

    def accept(table, key, value):
        if table is store.state and key == scoped:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
        return real_set(table, key, value)

    monkeypatch.setattr(type(store.state), "__setitem__", accept)
    task = asyncio.create_task(context.set_state_async("cursor", {"offset": 9}))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(store.close)
    reopened = SessionStore(path)
    try:
        assert read_plugin_state(reopened, scoped) == {"offset": 9}
    finally:
        reopened.close()


async def test_two_agents_and_mutable_payloads_stay_isolated(tmp_path):
    store = SessionStore(tmp_path / "shared.kohakutr")
    try:
        left, right = _context(store, "left"), _context(store, "right")
        payload = {"offset": 11}
        await left.set_state_async("cursor", payload)
        payload["offset"] = 99
        await right.set_state_async("cursor", {"offset": 22, "nested": []})
        loaded = await right.get_state_async("cursor")
        loaded["nested"].append(1)
        assert await left.get_state_async("cursor") == {"offset": 11}
        assert await right.get_state_async("cursor") == {"offset": 22, "nested": []}
        assert await _context(store, "left", "other").get_state_async("cursor") is None
    finally:
        store.close()


async def test_unattached_context_has_no_store_roundtrip():
    context = PluginContext(agent_name="a", _plugin_name="p")
    assert await context.get_state_async("cursor") is None
    assert await context.set_state_async("cursor", {"offset": 1}) is None


def test_sync_legacy_owner_scan_uses_meta_and_events(tmp_path):
    with ExitStack() as stack:
        unique = SessionStore(tmp_path / "unique.kohakutr")
        ambiguous = SessionStore(tmp_path / "ambiguous.kohakutr")
        stack.callback(unique.close)
        stack.callback(ambiguous.close)
        unique.append_event("solo", "user_message", {"content": "owned"})
        unique.state["plugin:shared:cursor"] = {"offset": 4}
        context = _context(unique, "solo")
        assert read_plugin_state(unique, context_state_key(context, "cursor")) == {
            "offset": 4
        }
        assert "plugin:shared:cursor" not in unique.state
        ambiguous.meta["agents"] = ["left", "right"]
        ambiguous.append_event("left", "user_message", {"content": "a"})
        ambiguous.append_event("right", "user_message", {"content": "b"})
        ambiguous.state["plugin:shared:cursor"] = "unknown"
        with pytest.raises(ConflictError, match="legacy"):
            write_plugin_state(
                ambiguous,
                context_state_key(_context(ambiguous, "left"), "cursor"),
                "replacement",
            )
        assert ambiguous.state["plugin:shared:cursor"] == "unknown"


def test_sync_methods_detach_mutable_values(tmp_path, request):
    store = SessionStore(tmp_path / "sync-copy.kohakutr")
    request.addfinalizer(store.close)
    context = _context(store, "only")
    payload = {"offset": 1}
    context.set_state("cursor", payload)
    payload["offset"] = 2
    loaded = context.get_state("cursor")
    assert loaded == {"offset": 1}
    loaded["offset"] = 3
    assert context.get_state("cursor") == {"offset": 1}
