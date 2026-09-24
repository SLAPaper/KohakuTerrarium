"""Negative controls for gaps not solved by composing owner and store queues."""

import asyncio
import threading
from types import SimpleNamespace

from kohakuterrarium.core.session_state_owner import SessionStateOwner
from kohakuterrarium.modules.plugin.base import PluginContext
from kohakuterrarium.modules.plugin.state_scope import context_state_key
from kohakuterrarium.session.store import SessionStore


async def test_legacy_sync_plugin_state_queues_behind_accepted_worker_work(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "plugin.kohakutr")
    entered, release = threading.Event(), threading.Event()
    accesses = []
    table_type = type(store.state)
    real_set = table_type.__setitem__

    def observed(table, key, value):
        if table is store.state:
            accesses.append(threading.get_ident())
        return real_set(table, key, value)

    def held_worker():
        entered.set()
        if not release.wait(5):
            raise TimeoutError("unreleased worker")

    try:
        monkeypatch.setattr(table_type, "__setitem__", observed)
        held = store.submit(held_worker)
        assert await asyncio.to_thread(entered.wait, 5)
        host = SimpleNamespace(session_store=store)
        context = PluginContext(
            agent_name="probe", _host_agent=host, _plugin_name="state"
        )
        scoped = context_state_key(context, "cursor")
        submit = store.submit

        def submit_and_release(fn, *args, **kwargs):
            future = submit(fn, *args, **kwargs)
            release.set()
            return future

        monkeypatch.setattr(store, "submit", submit_and_release)
        context.set_state("cursor", 7)
        assert context.get_state("cursor") == 7
        assert accesses == [store._affinity_worker_ident]
        assert accesses[0] != threading.get_ident()
        release.set()
        await asyncio.wrap_future(held)
        assert await store.run(lambda: store.state[scoped]) == 7
    finally:
        release.set()
        await asyncio.to_thread(store.close)


async def test_two_short_owner_callbacks_do_not_fence_intervening_conflicting_mutation(
    tmp_path,
):
    store = SessionStore(tmp_path / "fence.kohakutr")
    owner = SessionStateOwner(name="unfenced")
    owner.start()
    entered, release = threading.Event(), threading.Event()
    view = {"branch": 1}
    work = None

    async def unfenced_operation():
        selected = await owner.submit(lambda: view["branch"])

        def write():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("unreleased write")
            store.append_event("probe", "probe_selection", {"selected": selected})

        await store.run(write)
        await owner.submit_internal(lambda: view.update(branch=selected))

    try:
        work = asyncio.create_task(unfenced_operation())
        assert await asyncio.to_thread(entered.wait, 5)
        await owner.submit(lambda: view.update(branch=2))
        assert view["branch"] == 2
        release.set()
        await work
        assert view["branch"] == 1
        assert (await store.run(store.get_events, "probe"))[0]["selected"] == 1
    finally:
        release.set()
        if work is not None:
            await asyncio.gather(work, return_exceptions=True)
        await owner.close()
        await asyncio.to_thread(store.close)
