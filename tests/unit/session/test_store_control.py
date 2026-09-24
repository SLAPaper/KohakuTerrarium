"""Uncached control commits, readback, and ordering on the store worker."""

import asyncio
import threading

import pytest
from kohakuvault import KVault

from kohakuterrarium.errors import SessionError
from kohakuterrarium.session.store import SessionStore


def payload(target="openai/new"):
    return dict(
        schema=1,
        classification="portable",
        from_source={},
        to_source={},
        to_identifier=target,
    )


def _events(path, agent):
    reader = SessionStore(path)
    try:
        return reader.get_events(agent)
    finally:
        reader.close(update_status=False)


def test_negative_control_cached_append_can_survive_a_reported_flush_failure(
    tmp_path, monkeypatch
):
    store = SessionStore(
        tmp_path / "cached.kohakutr", flush_every_n_events=0, flush_every_n_seconds=0
    )
    try:

        def fail_flush():
            raise OSError("flush failed before commit")

        def cached_attempt():
            store.append_event("alice", "model_switch", payload())
            store.flush()

        with monkeypatch.context() as patch:
            patch.setattr(store, "flush", fail_flush)
            with pytest.raises(OSError, match="flush failed"):
                store.submit(cached_attempt).result(5)
        store.submit(store.flush).result(5)
        rows = store.submit(store.get_events, "alice").result(5)
        assert len(rows) == 1
        assert rows[0]["to_identifier"] == "openai/new"
    finally:
        store.close(update_status=False)


@pytest.mark.parametrize("after", [False, True])
def test_direct_write_failure_resolution(tmp_path, monkeypatch, after):
    path = tmp_path / "control.kohakutr"
    store = SessionStore(path, flush_every_n_events=0, flush_every_n_seconds=0)
    observed = []

    def observe(key, event):
        if event.get("type") != "model_switch":
            return
        reader = KVault(path, table="events")
        try:
            assert reader[key] == event
            observed.append(event["to_identifier"])
        finally:
            reader.close()

    store._event_subscribers.append(observe)
    store.submit(
        store.append_event, "host:alice", "user_message", {"content": "pending"}
    ).result(5)
    put = KVault.put

    def fail(writer, key, value, **kwargs):
        if isinstance(value, dict) and value.get("type") == "model_switch":
            assert observed == []
            if after:
                put(writer, key, value, **kwargs)
            raise OSError("direct put failed")
        return put(writer, key, value, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(KVault, "put", fail)
            call = store.submit(
                store.append_control_event, "host:alice", "model_switch", payload()
            )
            if after:
                call.result(5)
            else:
                with pytest.raises(OSError, match="direct put failed"):
                    call.result(5)
        assert observed == (["openai/new"] if after else [])
        reserved = store.submit(lambda: store._global_event_id).result(5)
        store.submit(
            store.append_event, "host:alice", "user_message", {"content": "later"}
        ).result(5)
        store.submit(store.events.flush_cache).result(5)
    finally:
        store.close(update_status=False)

    reopened = SessionStore(path)
    try:
        rows = reopened.get_events("host:alice")
        kinds = [row.get("type") for row in rows]
        assert kinds == (
            ["user_message", "model_switch", "user_message"]
            if after
            else ["user_message", "user_message"]
        )
        _, event_id = reopened.submit(
            reopened.append_control_event, "alice", "model_switch", payload()
        ).result(5)
        assert event_id > reserved
        reopened.submit(reopened.events.flush_cache).result(5)
    finally:
        reopened.close(update_status=False)
    assert [row.get("type") for row in _events(path, "alice")] == ["model_switch"]
    assert [row.get("type") for row in _events(path, "host:alice")] == (
        ["user_message", "model_switch", "user_message"]
        if after
        else ["user_message", "user_message"]
    )


@pytest.mark.parametrize("after", [False, True])
def test_counter_failure_prevents_control(tmp_path, monkeypatch, after):
    path = tmp_path / "control.kohakutr"
    store = SessionStore(path, flush_every_n_events=0, flush_every_n_seconds=0)
    put = KVault.put
    state_writes = []

    def fail(writer, key, value, **kwargs):
        if key == "counters:max_event_id" and isinstance(value, int):
            state_writes.append(value)
            if after:
                put(writer, key, value, **kwargs)
            raise OSError("reservation failed")
        return put(writer, key, value, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(KVault, "put", fail)
            with pytest.raises(OSError, match="reservation failed"):
                store.submit(
                    store.append_control_event, "alice", "model_switch", payload()
                ).result(5)
        assert (
            state_writes
        ), "counter reservation must be attempted before the event write"
        assert store.submit(lambda: store.get_events("alice")).result(5) == []
        cached = store.submit(lambda: "alice:e000000" in store.events).result(5)
        assert cached is False
    finally:
        store.close(update_status=False)
    assert _events(path, "alice") == []
    reopened = SessionStore(path)
    try:
        _, event_id = reopened.submit(
            reopened.append_control_event, "alice", "model_switch", payload()
        ).result(5)
        assert event_id > state_writes[-1] if after else event_id >= 1
    finally:
        reopened.close(update_status=False)


@pytest.mark.parametrize("outcome", ["unreadable", "mismatch"])
def test_unknown_commit_read_fails_closed(tmp_path, monkeypatch, outcome):
    store = SessionStore(tmp_path / "control.kohakutr")
    put = KVault.put
    get = KVault.get

    def fail_put(writer, key, value, **kwargs):
        if isinstance(value, dict) and value.get("type") == "model_switch":
            raise OSError("write outcome unknown")
        return put(writer, key, value, **kwargs)

    def fail_get(reader, key, *args, **kwargs):
        text = key.decode() if isinstance(key, bytes) else key
        if text == "alice:e000000":
            if outcome == "unreadable":
                raise OSError("read unavailable")
            return {"type": "model_switch", "to_identifier": "another payload"}
        return get(reader, key, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(KVault, "put", fail_put)
            patch.setattr(KVault, "get", fail_get)
            with pytest.raises(SessionError, match="Ambiguous model_switch"):
                store.submit(
                    store.append_control_event, "alice", "model_switch", payload()
                ).result(5)
    finally:
        store.close(update_status=False)


def test_cleanup_and_observer_errors_cannot_reject_confirmed_control(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "cleanup.kohakutr")
    close = KVault.close
    observed = []

    def fail_close(writer):
        close(writer)
        raise OSError("close failed after release")

    def fail_observer(key, event):
        raise OSError("observer failed")

    store.subscribe(fail_observer)
    store.subscribe(lambda key, event: observed.append(event["to_identifier"]))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(KVault, "close", fail_close)
            key, identity = store.submit(
                store.append_control_event, "alice", "model_switch", payload()
            ).result(5)
        assert observed == ["openai/new"]
        recorded = store.submit(store.get_events, "alice").result(5)
        assert len(recorded) == 1
        assert recorded[0]["event_id"] == identity
        assert (
            store.submit(lambda: store.events[key]["to_identifier"]).result(5)
            == "openai/new"
        )
    finally:
        store.close(update_status=False)


@pytest.mark.parametrize(
    "kind,readonly,error",
    [("text_chunk", False, ValueError), ("model_switch", True, SessionError)],
)
def test_invalid_control_request_leaves_counters_untouched(
    tmp_path, kind, readonly, error
):
    store = SessionStore(tmp_path / "refused.kohakutr")
    try:

        def attempt():
            store._readonly = readonly
            before = (store._global_event_id, dict(store._event_seq))
            with pytest.raises(error):
                store.append_control_event("alice", kind, payload())
            assert (store._global_event_id, store._event_seq) == before
            assert store.get_events("alice") == []

        store.submit(attempt).result(5)
    finally:
        store.close(update_status=False)


def test_callback_cannot_mutate_persisted_and_sees_commit(tmp_path):
    path = tmp_path / "control.kohakutr"
    store = SessionStore(path)
    seen = {}

    def observe(key, event):
        reader = KVault(path, table="events")
        try:
            committed = dict(reader[key])
        finally:
            reader.close()
        event["to_identifier"] = "mutated"
        event["injected"] = True
        seen["key"] = key
        seen["callback"] = event["to_identifier"]
        seen["committed"] = committed["to_identifier"]
        seen["injected_absent"] = "injected" not in committed

    store._event_subscribers.append(observe)
    original = payload()
    try:
        key, _ = store.submit(
            store.append_control_event, "alice", "model_switch", original
        ).result(5)
        assert original["to_identifier"] == "openai/new"
        assert "type" not in original
        assert seen == {
            "key": key,
            "callback": "mutated",
            "committed": "openai/new",
            "injected_absent": True,
        }
        again = KVault(path, table="events")
        try:
            assert again[key]["to_identifier"] == "openai/new"
            assert "injected" not in again[key]
        finally:
            again.close()
    finally:
        store.close(update_status=False)


def test_direct_offworker_refusal_has_no_side_effects(tmp_path):
    path = tmp_path / "control.kohakutr"
    store = SessionStore(path)
    try:
        before = store.submit(
            lambda: (store._global_event_id, dict(store._event_seq))
        ).result(5)
        with pytest.raises(RuntimeError, match="affinity worker"):
            store.append_control_event("alice", "model_switch", payload())
        after = store.submit(
            lambda: (store._global_event_id, dict(store._event_seq))
        ).result(5)
        assert after == before
        assert store.submit(lambda: store.get_events("alice")).result(5) == []
    finally:
        store.close(update_status=False)
    assert _events(path, "alice") == []


def test_fifo_shared_namespaces_and_thread_ids(tmp_path):
    path = tmp_path / "fifo.kohakutr"
    store = SessionStore(path, flush_every_n_events=0, flush_every_n_seconds=0)
    worker = {}

    def record():
        worker["ident"] = threading.get_ident()
        first = store.append_event("host:alice", "user_message", {"content": "n1"})
        control = store.append_control_event(
            "host:alice", "model_switch", payload("openai/mid")
        )
        sibling = store.append_control_event(
            "alice", "model_switch", payload("openai/other")
        )
        last = store.append_event("host:alice", "user_message", {"content": "n2"})
        return first, control, sibling, last

    try:
        n1, control, sibling, n2 = store.submit(record).result(5)
        assert worker["ident"] == store._affinity_worker_ident
        assert worker["ident"] != threading.get_ident()
        assert [item[1] for item in (n1, control, sibling, n2)] == sorted(
            item[1] for item in (n1, control, sibling, n2)
        )
        assert len({item[1] for item in (n1, control, sibling, n2)}) == 4
        assert control[0] == "host:alice:e000001"
        assert sibling[0] == "alice:e000000"
        rows = store.submit(lambda: store.get_events("host:alice")).result(5)
        assert [
            row["content"] if row["type"] == "user_message" else row["to_identifier"]
            for row in rows
        ] == [
            "n1",
            "openai/mid",
            "n2",
        ]
    finally:
        store.close(update_status=False)


@pytest.mark.parametrize("cancel_when", ["queued", "running"])
def test_cancelled_caller_still_commits_and_close_drains(tmp_path, cancel_when):
    path = tmp_path / "cancel.kohakutr"
    store = SessionStore(path)
    release = threading.Event()
    started = threading.Event()

    def write():
        if cancel_when == "running":
            started.set()
            assert release.wait(5)
        return store.append_control_event(
            "alice", "model_switch", payload("openai/kept")
        )

    async def cancel_accepted():
        if cancel_when == "queued":
            # Block the worker so the control call is accepted but not started.
            store.submit(release.wait, 5)
        task = asyncio.create_task(store.run(write))
        await asyncio.sleep(0)
        if cancel_when == "running":
            assert await asyncio.to_thread(started.wait, 2)
        later = store.submit(
            store.append_event, "alice", "user_message", {"content": "after"}
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert not release.is_set()
        assert not later.done()
        return later

    later = None
    try:
        later = asyncio.run(cancel_accepted())
    finally:
        release.set()
        if later is not None:
            later.result(5)
        store.close(update_status=False)
    rows = _events(path, "alice")
    assert [row.get("type") for row in rows] == ["model_switch", "user_message"]
    assert rows[0]["to_identifier"] == "openai/kept"
    assert [row["event_id"] for row in rows] == [1, 2]
