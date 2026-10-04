"""Critical text persistence must fence synchronous handoff and reject unproven streams."""

from types import SimpleNamespace

import pytest

from kohakuterrarium.errors import SessionError
from kohakuterrarium.session.output import SessionOutput
from kohakuterrarium.session.store import SessionStore


async def test_sync_flush_surfaces_critical_append_failure_and_keeps_durable_slot(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "critical.kohakutr")
    agent = SimpleNamespace(_turn_index=1, _branch_id=1, _parent_branch_path=[])
    sink = SessionOutput("alice", store, agent)
    append = SessionStore.append_event
    try:
        await sink._begin_generation(
            {"schema": 1, "generation_id": "generation-one", "source": {}}
        )
        await sink.write_stream("partial" * 100)
        await sink.drain()
        before = store.submit(store.state.get, "alice:open_text").result()

        def fail(candidate, namespace, kind, data, **kwargs):
            if candidate is store and kind == "text_chunk":
                raise OSError("critical text append failed")
            return append(candidate, namespace, kind, data, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(SessionStore, "append_event", fail)
            with pytest.raises(OSError, match="critical text"):
                sink.flush_sync()
        assert store.submit(store.state.get, "alice:open_text").result() == before
        assert not [
            r
            for r in store.submit(store.get_events, "alice").result()
            if r["type"] == "text_chunk"
        ]
    finally:
        store.close(update_status=False)


async def test_unbound_assistant_stream_cannot_create_a_new_unresumable_slot(tmp_path):
    store = SessionStore(tmp_path / "unbound.kohakutr")
    sink = SessionOutput("alice", store, None)
    try:
        await sink.drain()
        with pytest.raises(SessionError, match="generation"):
            await sink.write_stream("unproven" * 100)
        assert sink._open_text._buf == ""
        assert store.submit(store.state.get, "alice:open_text").result() in (None, "")
        assert store.submit(store.get_events, "alice").result() == []
    finally:
        store.close(update_status=False)
