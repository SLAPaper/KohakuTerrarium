"""Graph checkpoint capture stays on-loop; detached metadata IO stays on its worker."""

import asyncio
import threading
from dataclasses import replace

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.errors import GraphManifestPersistenceError
from kohakuterrarium.bootstrap import agent_init, llm
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.terrarium import (
    graph_checkpoint,
    graph_manifest,
    topology_snapshot,
)
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("failure", [False, "io", "manifest", "cancel"])
async def test_checkpoint_worker_barrier_keeps_capture_detached_and_errors_visible(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(
        agent_init, "create_llm_provider", lambda *a, **k: ScriptedLLM(["ok"])
    )
    monkeypatch.setattr(llm, "create_llm_provider", lambda *a, **k: ScriptedLLM(["ok"]))
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), io="headless", start=True
        )
        gid, store = creature.graph_id, creature.agent.session_store
        persist = graph_checkpoint._persist_checkpoint

        def held(*args):
            assert threading.get_ident() == store._affinity_worker_ident
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            if failure == "io":
                raise OSError("checkpoint unavailable")
            if failure == "manifest":
                raise GraphManifestPersistenceError("checkpoint unavailable")
            return persist(*args)

        with monkeypatch.context() as patch:
            patch.setattr(graph_checkpoint, "_persist_checkpoint", held)
            pending = asyncio.create_task(
                engine.add_channel(gid, "detached", description="captured")
            )
            try:
                await asyncio.wait_for(entered.wait(), 3)
                channels = engine._topology.graphs[gid].channels
                channels["detached"] = replace(
                    channels["detached"], description="changed after capture"
                )
                assert not pending.done()
                if failure == "cancel":
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert gid in graph_checkpoint.dirty_graphs(engine)
                    release.set()
                elif failure:
                    release.set()
                    with pytest.raises(
                        GraphManifestPersistenceError, match="checkpoint unavailable"
                    ) as exc:
                        await pending
                    assert exc.value.applied
                    assert gid in graph_checkpoint.dirty_graphs(engine)
                else:
                    release.set()
                    await pending
                    saved = await store.run(graph_manifest.load_manifest, store)
                    assert saved.channels[0].description == "captured"
                    meta = await store.run(store.load_meta)
                    assert (
                        meta[topology_snapshot.META_KEY]["channels"][0]["description"]
                        == "captured"
                    )
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)
        assert await graph_checkpoint.checkpoint(engine, gid)
        saved = await store.run(graph_manifest.load_manifest, store)
        assert saved.channels[0].description == "changed after capture"
        assert gid not in graph_checkpoint.dirty_graphs(engine)


async def test_checkpoint_reads_and_writes_only_on_store_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        agent_init, "create_llm_provider", lambda *a, **k: ScriptedLLM(["ok"])
    )
    monkeypatch.setattr(llm, "create_llm_provider", lambda *a, **k: ScriptedLLM(["ok"]))
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), io="headless", start=True
        )
        store, gid = creature.agent.session_store, creature.graph_id
        capture = graph_manifest.capture_manifest
        captured_threads, accesses = [], []

        def captured(*args, **kwargs):
            captured_threads.append(threading.get_ident())
            return capture(*args, **kwargs)

        owner_thread = threading.get_ident()
        original = store.meta

        class WorkerMeta:
            def __getattr__(self, name):
                return getattr(original, name)

            def get(self, *args):
                assert threading.get_ident() == store._affinity_worker_ident
                accesses.append("get")
                return original.get(*args)

            def __setitem__(self, key, value):
                assert threading.get_ident() == store._affinity_worker_ident
                accesses.append(key)
                original[key] = value

        with monkeypatch.context() as patch:
            patch.setattr(store, "meta", WorkerMeta())
            patch.setattr(graph_manifest, "capture_manifest", captured)
            assert await graph_checkpoint.checkpoint(engine, gid)
        assert captured_threads == [owner_thread]
        assert graph_manifest.MANIFEST_KEY in accesses
        assert topology_snapshot.META_KEY in accesses
        saved = await store.run(graph_manifest.load_manifest, store)
        assert [row.name for row in saved.creatures] == ["alice"]
        revision = saved.revision
        await engine.add_channel(
            gid, "new-channel", description="persisted description"
        )
        saved = await store.run(graph_manifest.load_manifest, store)
        assert saved.revision == revision + 1
        assert [(row.name, row.description) for row in saved.channels] == [
            ("new-channel", "persisted description")
        ]
