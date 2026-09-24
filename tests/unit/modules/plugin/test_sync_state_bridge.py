"""Legacy synchronous state calls acknowledge canonical worker operations."""

import threading
from types import SimpleNamespace

import pytest

from kohakuterrarium.modules.plugin.base import PluginContext
from kohakuterrarium.modules.plugin.state_io import plugin_state_binding
from kohakuterrarium.modules.plugin.state_scope import context_state_key
from kohakuterrarium.session.store import SessionStore


@pytest.fixture
def scoped(tmp_path):
    store = SessionStore(tmp_path / "sync.kohakutr")
    host = SimpleNamespace(config=SimpleNamespace(name="alice"), session_store=store)
    context = PluginContext(_host_agent=host, _plugin_name="legacy")
    try:
        yield store, context
    finally:
        store.close()


async def test_sync_loop_call_uses_worker_and_detaches_values(scoped, monkeypatch):
    store, ctx = scoped
    threads = []
    original = type(store.state).__setitem__

    def observed(table, key, value):
        if table is store.state:
            threads.append(threading.get_ident())
        return original(table, key, value)

    monkeypatch.setattr(type(store.state), "__setitem__", observed)
    payload = {"items": [1]}
    assert ctx.set_state("value", payload) is None
    payload["items"].append(2)
    loaded = ctx.get_state("value")
    assert loaded == {"items": [1]}
    loaded["items"].append(3)
    assert ctx.get_state("value") == {"items": [1]}
    assert threads == [store._affinity_worker_ident]
    assert threads[0] != threading.get_ident()


def test_sync_error_propagates_without_success(scoped, monkeypatch):
    store, ctx = scoped
    original = type(store.state).__setitem__
    key = context_state_key(ctx, "value")

    def fail(table, name, value):
        if table is store.state and name == key:
            raise OSError("sync write rejected")
        return original(table, name, value)

    monkeypatch.setattr(type(store.state), "__setitem__", fail)
    with pytest.raises(OSError, match="sync write rejected"):
        ctx.set_state("value", 7)
    assert ctx.get_state("value") is None


async def test_sync_call_already_on_worker_does_not_queue_itself(scoped, monkeypatch):
    store, ctx = scoped
    submit = store.submit

    def reject_self_queue(fn, *args, **kwargs):
        if threading.get_ident() == store._affinity_worker_ident:
            raise AssertionError("worker queued a synchronous wait onto itself")
        return submit(fn, *args, **kwargs)

    monkeypatch.setattr(store, "submit", reject_self_queue)
    await store.run(ctx.set_state, "value", 11)
    assert await store.run(ctx.get_state, "value") == 11


def test_accepted_worker_does_not_wait_for_coordinator_binding_lock(scoped):
    store, ctx = scoped
    with plugin_state_binding(ctx.host_agent):
        store.submit(ctx.set_state, "value", 19).result(timeout=3)
        assert store.submit(ctx.get_state, "value").result(timeout=3) == 19


def test_sync_bridge_rejects_closed_store(scoped):
    store, ctx = scoped
    ctx.set_state("value", 1)
    store.close()
    assert store._affinity_worker_ident is None
    with pytest.raises(RuntimeError, match="closed"):
        ctx.set_state("value", 2)
    with pytest.raises(RuntimeError, match="closed"):
        ctx.get_state("value")


def test_accepted_worker_sync_call_completes_during_close_drain(scoped, monkeypatch):
    store, ctx = scoped
    entered, closing = threading.Event(), threading.Event()
    shutdown = store._shutdown_affinity

    def accepted():
        entered.set()
        assert closing.wait(3)
        ctx.set_state("value", 17)
        return ctx.get_state("value")

    def begin_drain():
        assert store._closed
        closing.set()
        shutdown()

    monkeypatch.setattr(store, "_shutdown_affinity", begin_drain)
    future = store.submit(accepted)
    assert entered.wait(3)
    store.close()
    assert future.result() == 17
    reopened = SessionStore(store.path)
    try:
        assert reopened.state[context_state_key(ctx, "value")] == 17
    finally:
        reopened.close()


async def test_sync_and_async_calls_share_fifo(scoped):
    store, ctx = scoped
    first = store.submit(
        lambda: store.state.__setitem__(context_state_key(ctx, "value"), 3)
    )
    assert ctx.get_state("value") == 3
    assert first.done()
    await ctx.set_state_async("value", 4)
    assert ctx.get_state("value") == 4
    ctx.set_state("value", 5)
    assert await ctx.get_state_async("value") == 5
