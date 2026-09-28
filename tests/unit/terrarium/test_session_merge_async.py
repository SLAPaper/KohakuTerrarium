"""Prepared merge failures preserve membership and source retirement responsibility."""

from pathlib import Path

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium import (
    channels,
    channel_lifecycle,
    session_coord,
    session_merge_async,
    topology,
    topology_admission,
)


async def _member(engine, name):
    return await engine.add_creature(
        AgentConfig(name=name), llm=ScriptedLLM([name]), io="headless", start=True
    )


@pytest.mark.parametrize("cleanup", ["close", "archive", "owner"])
async def test_abort_cleanup_failure_preserves_import_error_and_owned_handles(
    tmp_path, monkeypatch, cleanup
):
    targets, attempts = [], []
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice, bob = await _member(engine, "alice"), await _member(engine, "bob")
        if cleanup == "owner":
            peer = await _member(engine, "peer")
            await engine.connect(alice, peer, channel="existing")
        sources = {alice.agent.session_store, bob.agent.session_store}
        import_data = session_coord.install_runtime_transfer
        submit = topology_admission.submit_internal_session_state
        close = type(alice.agent.session_store).close

        def failed_import(store, captured):
            targets.append(store)
            import_data(store, captured)
            raise OSError("primary import failure")

        def failed_close(store, **kwargs):
            if store in targets:
                raise RuntimeError("secondary close failure")
            return close(store, **kwargs)

        def failed_archive(store):
            raise RuntimeError("secondary archive failure")

        async def failed_owner(agent, callback):
            if targets:
                attempts.append(agent)
                if agent is alice.agent:
                    raise RuntimeError("secondary owner failure")
            return await submit(agent, callback)

        with monkeypatch.context() as patch:
            patch.setattr(session_coord, "install_runtime_transfer", failed_import)
            if cleanup == "close":
                patch.setattr(type(alice.agent.session_store), "close", failed_close)
            elif cleanup == "archive":
                patch.setattr(
                    session_merge_async, "archive_unpublished_store", failed_archive
                )
            else:
                patch.setattr(
                    topology_admission, "submit_internal_session_state", failed_owner
                )
            with pytest.raises(OSError, match="primary import failure"):
                await engine.connect(alice, bob, channel="failing")
        assert targets
        assert all(not store._closed for store in sources)
        if cleanup == "archive":
            assert targets[0]._closed and Path(targets[0].path).exists()
        elif cleanup == "close":
            assert targets[0] in engine._retained_session_stores
        else:
            assert bob.agent in attempts and bob.agent._session_attach_uncertain
            assert sources <= engine._retained_session_stores
    assert all(store._closed for store in targets)


@pytest.mark.parametrize("operation", ["merge", "split"])
async def test_post_install_publication_failure_fences_and_retains_bindings(
    tmp_path, monkeypatch, operation
):
    targets = []
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice, bob = await _member(engine, "alice"), await _member(engine, "bob")
        if operation == "split":
            await engine.connect(alice, bob, channel="bridge")
        sources = {alice.agent.session_store, bob.agent.session_store}

        def fail(*args, **kwargs):
            targets.extend([alice.agent.session_store, bob.agent.session_store])
            raise OSError("publication failed after owner binding")

        with monkeypatch.context() as patch:
            if operation == "merge":
                patch.setattr(channels, "_merge_environment_into", fail)
                pending = engine.connect(alice, bob, channel="bridge")
            else:
                patch.setattr(channel_lifecycle, "remove_channel_trigger", fail)
                pending = engine.disconnect(alice, bob, channel="bridge")
            with pytest.raises(OSError, match="publication failed"):
                await pending
        assert (
            alice.agent._session_attach_uncertain
            and bob.agent._session_attach_uncertain
        )
        assert all(not store._closed for store in sources)
        assert targets and all(not store._closed for store in targets)
        for store in targets:
            assert (
                store in engine._session_stores.values()
                or store in engine._retained_session_stores
            )
    assert all(store._closed for store in targets)


async def test_fresh_merge_and_split_preserve_durable_session_identity(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice, bob = await _member(engine, "alice"), await _member(engine, "bob")
        source_ids = {
            store.session_id
            for store in (alice.agent.session_store, bob.agent.session_store)
        }
        await engine.connect(alice, bob, channel="bridge")
        merged = alice.agent.session_store
        meta = await merged.run(merged.load_meta)
        assert meta["session_id"] == merged.session_id
        assert set(meta["parent_session_ids"]) == source_ids
        merged_id = meta["session_id"]
        await engine.disconnect(alice, bob, channel="bridge")
        for creature in (alice, bob):
            store = creature.agent.session_store
            meta = await store.run(store.load_meta)
            assert meta["session_id"] == store.session_id
            assert merged_id in meta["parent_session_ids"]


@pytest.mark.parametrize("reused", [False, True])
async def test_prepare_import_failure_keeps_original_topology_and_fences_mutated_target(
    tmp_path, monkeypatch, reused
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice, bob = await _member(engine, "alice"), await _member(engine, "bob")
        if reused:
            peer = await _member(engine, "peer")
            await engine.connect(alice, peer, channel="existing")
        before = alice.graph_id, bob.graph_id
        sources = alice.agent.session_store, bob.agent.session_store
        for store in sources:
            await store.run(
                store.write_artifact, "retained.bin", b"failed-copy-evidence"
            )
        original = session_coord.install_runtime_transfer
        targets = []

        def failed(store, captured):
            targets.append(store)
            original(store, captured)
            raise OSError("target import failed after write")

        with monkeypatch.context() as patch:
            patch.setattr(session_coord, "install_runtime_transfer", failed)
            with pytest.raises(OSError, match="target import failed"):
                await engine.connect(alice, bob, channel="joining")
        assert (alice.graph_id, bob.graph_id) == before
        assert all(not store._closed for store in sources)
        assert (
            alice.agent.session_store is sources[0]
            and bob.agent.session_store is sources[1]
        )
        if reused:
            assert (
                alice.agent._session_attach_uncertain
                and bob.agent._session_attach_uncertain
            )
            assert not targets[0]._closed
        else:
            assert targets[0]._closed
            assert not getattr(alice.agent, "_session_attach_uncertain", False)
            await engine.connect(alice, bob, channel="retry")
            assert alice.graph_id == bob.graph_id
            assert alice.agent.session_store is not targets[0]
            archived = list(
                (tmp_path / "sessions" / ".failed-handoffs").glob(
                    "*/" + Path(targets[0].path).name
                )
            )
            assert len(archived) == 1 and archived[0].is_file()
            assert (
                archived[0].with_name(f"{archived[0].stem}.artifacts") / "retained.bin"
            ).read_bytes() == b"failed-copy-evidence"


async def test_kept_store_identity_does_not_depend_on_delta_source_order(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice, peer, bob = [
            await _member(engine, name) for name in ("alice", "peer", "bob")
        ]
        await engine.connect(alice, peer, channel="existing")
        kept = alice.agent.session_store
        await kept.run(
            lambda: kept.meta.__setitem__("conversation_id", "kept-conversation")
        )
        await bob.agent.session_store.run(
            lambda: bob.agent.session_store.meta.__setitem__(
                "conversation_id", "dropped-conversation"
            )
        )
        merge = topology._merge_graphs

        def reversed_sources(*args):
            delta = merge(*args)
            delta.old_graph_ids.reverse()
            return delta

        monkeypatch.setattr(topology, "_merge_graphs", reversed_sources)
        await engine.connect(alice, bob, channel="joining")
        assert alice.agent.session_store is kept
        assert (
            await kept.run(lambda: kept.meta["conversation_id"]) == "kept-conversation"
        )


async def test_retirement_failure_is_retained_after_successful_owner_install(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice, bob = await _member(engine, "alice"), await _member(engine, "bob")
        source = alice.agent.session_store
        original = session_merge_async._retire_metadata

        def failed(store):
            if store is source:
                raise OSError("retirement unavailable")
            return original(store)

        with monkeypatch.context() as patch:
            patch.setattr(session_merge_async, "_retire_metadata", failed)
            await engine.connect(alice, bob, channel="joining")
        assert alice.graph_id == bob.graph_id
        assert alice.agent.session_store is bob.agent.session_store is not source
        assert source in engine._retained_session_stores and not source._closed
    assert source._closed
