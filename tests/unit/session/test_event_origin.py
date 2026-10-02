"""Persistent event provenance survives retries, hops, reopen, and forks."""

from copy import deepcopy
from uuid import uuid4

import pytest

import kohakuterrarium.session.store as store_module
from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.event_origin import (
    EVENT_ORIGIN,
    STORE_UUID,
    append_mirrored,
    capture_events,
    import_events,
    store_uuid,
)
from kohakuterrarium.session.history_replay import replay_selected_events
from kohakuterrarium.session.history_support import dedupe_adjacent_duplicate_events
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.terrarium.session_coord import copy_events_into


@pytest.fixture
def stores(tmp_path):
    handles = [SessionStore(tmp_path / f"{name}.kohakutr") for name in ("a", "b", "c")]
    yield handles
    for store in handles:
        store.close(update_status=False)


def append(store, name="alice", text="same"):
    return store.submit(
        store.append_event, name, "user_message", {"content": text}
    ).result()


def test_capture_and_replay_keep_numeric_physical_order_past_six_digits(stores):
    a, b, _ = stores
    a.submit(lambda: a._event_seq.update(alice=999999)).result()
    append(a, text="before rollover")
    append(a, text="after rollover")
    _, captured = a.submit(capture_events, a).result()
    assert [row[2]["content"] for row in captured] == [
        "before rollover",
        "after rollover",
    ]
    assert [row["content"] for row in a.submit(a.get_events, "alice").result()] == [
        "before rollover",
        "after rollover",
    ]
    copy_events_into(a, b)
    assert [row["content"] for row in b.submit(b.get_events, "alice").result()] == [
        "before rollover",
        "after rollover",
    ]


def test_replay_keeps_independent_origins_with_identical_user_coordinates(stores):
    a, b, destination = stores
    for source in (a, b):
        source.submit(
            source.append_event,
            "alice",
            "user_message",
            {"content": "same", "turn_index": 1, "branch_id": 1},
        ).result()
        copy_events_into(source, destination)
    rows = destination.submit(destination.get_events, "alice").result()
    messages = replay_selected_events(rows, include_metadata=True)
    assert [message["content"] for message in messages] == ["same", "same"]
    assert [message["metadata"]["event_id"] for message in messages] == [
        row["event_id"] for row in rows
    ]
    destination.close(update_status=False)
    reopened = SessionStore(destination.path)
    try:
        assert (
            replay_selected_events(
                reopened.submit(reopened.get_events, "alice").result(),
                include_metadata=True,
            )
            == messages
        )
    finally:
        reopened.close(update_status=False)


def test_replay_collapses_nonadjacent_copies_by_origin():
    origin = [str(uuid4()), "alice", 1]
    first = {
        "type": "text_chunk",
        "event_id": 1,
        "content": "A",
        "ts": 10,
        EVENT_ORIGIN: origin,
    }
    middle = {"type": "text_chunk", "event_id": 2, "content": "B"}
    duplicate = {**first, "event_id": 3}
    assert dedupe_adjacent_duplicate_events([first, middle, duplicate]) == [
        first,
        middle,
    ]
    assert replay_selected_events([first, middle, duplicate]) == [
        {"role": "assistant", "content": "AB"}
    ]


@pytest.mark.parametrize(
    "field,value", [("content", "different"), ("ts", 11), ("branch_id", 2)]
)
def test_replay_rejects_conflicting_payload_for_one_origin(field, value):
    first = {
        "type": "user_message",
        "event_id": 1,
        "content": "same",
        "ts": 10,
        "turn_index": 1,
        "branch_id": 1,
        EVENT_ORIGIN: [str(uuid4()), "alice", 1],
    }
    second = {**first, "event_id": 2, field: value}
    with pytest.raises(HistoryCompatibilityError, match="origin"):
        replay_selected_events([first, second])


@pytest.mark.parametrize(
    "origin",
    [
        None,
        [],
        ["bad", "alice", 1],
        [str(uuid4()), "alice", True],
        [str(uuid4()), "", 1],
    ],
)
def test_replay_rejects_malformed_origin(origin):
    with pytest.raises(HistoryCompatibilityError, match="origin"):
        replay_selected_events(
            [
                {
                    "type": "user_message",
                    "event_id": 1,
                    "content": "x",
                    EVENT_ORIGIN: origin,
                }
            ]
        )


@pytest.mark.parametrize("marked_first", [False, True])
def test_replay_does_not_equate_marked_and_legacy_users(marked_first):
    legacy = {
        "type": "user_message",
        "event_id": 1,
        "turn_index": 1,
        "branch_id": 1,
        "content": "same",
    }
    marked = {**legacy, "event_id": 2, EVENT_ORIGIN: [str(uuid4()), "alice", 7]}
    events = [marked, legacy] if marked_first else [legacy, marked]
    assert [
        m["metadata"]["event_id"]
        for m in replay_selected_events(events, include_metadata=True)
    ] == [e["event_id"] for e in events]


def test_origin_dedup_keeps_branch_selection_and_does_not_mutate_inputs():
    events = [
        {
            "type": "user_message",
            "event_id": i,
            "turn_index": 1,
            "branch_id": branch,
            "content": str(branch),
            EVENT_ORIGIN: [str(uuid4()), "alice", i],
        }
        for i, branch in [(1, 1), (2, 2)]
    ]
    original = deepcopy(events)
    assert replay_selected_events(events, branch_view={1: 1}) == [
        {"role": "user", "content": "1"}
    ]
    assert events == original


def test_identity_flush_failure_prevents_export_and_retry_reflushes(
    stores, monkeypatch
):
    a, b, _ = stores
    append(a)
    flush = type(a.meta).flush_cache
    attempts = []

    def held_flush(table):
        if table is a.meta:
            attempts.append(True)
            raise OSError("identity flush failed")
        return flush(table)

    with monkeypatch.context() as patch:
        patch.setattr(type(a.meta), "flush_cache", held_flush)
        for _ in range(2):
            with pytest.raises(OSError, match="identity flush failed"):
                copy_events_into(a, b)
            assert b.submit(b.get_events, "alice").result() == []
    assert len(attempts) == 2
    assert copy_events_into(a, b) == 1


def test_identity_enabled_reopen_rejects_unreadable_row(stores, monkeypatch):
    a, _, _ = stores
    append(a)
    a.submit(capture_events, a).result()
    events_type = type(a.events)
    path = a.path
    a.close(update_status=False)
    get = events_type.__getitem__

    def fail_event(table, key):
        if key == b"alice:e000000" or key == "alice:e000000":
            raise OSError("row unreadable")
        return get(table, key)

    with monkeypatch.context() as patch:
        patch.setattr(events_type, "__getitem__", fail_event)
        with pytest.raises(OSError, match="row unreadable"):
            opened = SessionStore(path)
            opened.close(update_status=False)


@pytest.mark.parametrize("change", ["content", "namespace", "origin"])
def test_mirror_conflicting_preset_id_rejects_before_append(stores, change):
    a, _, _ = stores
    a.submit(store_uuid, a).result()
    data = {"event_id": 4, "content": "original", "ts": 100}
    a.submit(append_mirrored, a, "alice", "message", deepcopy(data)).result()
    before = a.submit(a.get_events, "alice").result()
    if change == "content":
        data["content"] = "different"
    if change == "origin":
        data[EVENT_ORIGIN] = [str(uuid4()), "alice", 8]
    name = "other" if change == "namespace" else "alice"
    with pytest.raises(ValueError, match="Conflicting mirrored"):
        a.submit(append_mirrored, a, name, "message", data).result()
    assert a.submit(a.get_events, "alice").result() == before
    assert a.submit(a.get_events, "other").result() == []


def test_direct_preset_id_collision_cannot_corrupt_activated_store(stores):
    a, b, _ = stores
    _, eid = append(a)
    copy_events_into(a, b)
    with pytest.raises(ValueError, match="already exists"):
        a.submit(
            a.append_event,
            "alice",
            "message",
            {"event_id": eid, "content": "collision"},
        ).result()
    before = b.submit(b.get_events, "alice").result()
    assert copy_events_into(a, b) == 0
    assert b.submit(b.get_events, "alice").result() == before


def test_multihop_receipts_and_return_to_source(stores):
    a, b, c = stores
    _, original_id = append(a)
    captured = a.submit(capture_events, a).result()
    first = b.submit(import_events, b, captured).result()
    second = c.submit(import_events, c, b.submit(capture_events, b).result()).result()
    assert (
        first[0]["origin"] == second[0]["origin"] == (captured[0], "alice", original_id)
    )
    assert second[0]["source"][0] != captured[0]
    assert copy_events_into(c, a) == 0
    assert len(a.submit(a.get_events, "alice").result()) == 1
    repeat = c.submit(import_events, c, captured).result()
    assert not repeat[0]["inserted"]
    assert repeat[0]["destination"] == second[0]["destination"]


def test_rename_reopen_preserves_identity(stores, tmp_path):
    a, b, _ = stores
    append(a)
    copy_events_into(a, b)
    uid = a.submit(store_uuid, a).result()
    path = a.path
    a.close(update_status=False)
    renamed = tmp_path / "renamed.kohakutr"
    from_path = tmp_path / "a.kohakutr"
    assert str(from_path) == path
    from_path.rename(renamed)
    reopened = SessionStore(renamed)
    try:
        assert reopened.submit(store_uuid, reopened).result() == uid
        copy_events_into(reopened, b)
        assert len(b.submit(b.get_events, "alice").result()) == 1
    finally:
        reopened.close(update_status=False)


@pytest.mark.parametrize("mutation", [None, "same", "change", "drop"])
def test_fork_physical_identity_and_shared_prefix(stores, tmp_path, mutation):
    a, b, _ = stores
    append(a, text="first")
    _, eid = append(a, text="second")
    copy_events_into(a, b)

    def mutate(row):
        if mutation == "drop":
            return None
        if mutation == "change":
            row["content"] = "changed"
        row[EVENT_ORIGIN] = [str(uuid4()), "alice", 999]
        return row

    child = a.fork(
        str(tmp_path / "fork.kohakutr"),
        at_event_id=eid,
        mutate=mutate if mutation else None,
    )
    try:
        assert child.meta[STORE_UUID] != a.meta[STORE_UUID]
        copy_events_into(child, b)
        values = [row["content"] for row in b.submit(b.get_events, "alice").result()]
        assert values == (
            ["first", "second", "changed"]
            if mutation == "change"
            else ["first", "second"]
        )
    finally:
        child.close(update_status=False)


@pytest.mark.parametrize(
    "damage",
    ["uuid", "shape", "namespace", "id", "duplicate", "conflict", "unactivated"],
)
def test_invalid_identity_fails_before_destination_append(stores, damage):
    a, b, _ = stores
    append(a)
    copy_events_into(a, b)
    before = deepcopy(b.submit(b.get_events, "alice").result())
    if damage == "uuid":
        a.submit(lambda: a.meta.__setitem__(STORE_UUID, "invalid")).result()
    elif damage == "conflict":
        row = a.submit(a.get_events, "alice").result()[0]
        row["content"] = "changed"
        a.submit(lambda: a.events.__setitem__("alice:e000000", row)).result()
    else:
        row = deepcopy(before[0])
        if damage == "shape":
            row[EVENT_ORIGIN] = []
        if damage == "namespace":
            row[EVENT_ORIGIN][1] = "other"
        if damage == "id":
            row[EVENT_ORIGIN][2] = True
        if damage == "duplicate":
            row["event_id"] = 99
        key = "alice:e000001" if damage == "duplicate" else "alice:e000000"
        b.submit(lambda: b.events.__setitem__(key, row)).result()
        if damage == "unactivated":
            b.submit(lambda: b.meta.__delitem__(STORE_UUID)).result()
        before = deepcopy(b.submit(b.get_events, "alice").result())
    with pytest.raises(ValueError):
        copy_events_into(a, b)
    assert b.submit(b.get_events, "alice").result() == before


def test_reserved_origin_cannot_be_supplied_to_public_append(stores):
    a, _, _ = stores
    payload = {EVENT_ORIGIN: [str(uuid4()), "alice", 1], "content": "collision"}
    with pytest.raises(ValueError, match="Reserved"):
        a.submit(a.append_event, "alice", "message", payload).result()
    assert a.submit(a.get_events, "alice").result() == []


@pytest.mark.parametrize("identity", [False, True])
def test_readonly_source_never_mints_identity(stores, identity):
    a, b, _ = stores
    append(a)
    if identity:
        a.submit(store_uuid, a).result()
    path = a.path
    a.close(update_status=False)
    source = SessionStore.open_readonly(path)
    try:
        if identity:
            assert copy_events_into(source, b) == 1
        else:
            with pytest.raises(ValueError, match="Read-only"):
                copy_events_into(source, b)
            assert source.meta.get(STORE_UUID) is None
        with pytest.raises(ValueError, match="read-only"):
            copy_events_into(b, source)
    finally:
        source.close(update_status=False)


@pytest.mark.parametrize("persisted_max", [0, 100])
def test_all_namespaces_and_stale_counter_reopen(stores, monkeypatch, persisted_max):
    a, b, _ = stores
    for namespace in ("terrarium", "alice:attached:peer", "alice"):
        append(a, name=namespace)
    assert copy_events_into(a, b) == 3
    b.close(update_status=False)
    corrupt = SessionStore(b.path)
    corrupt.state["counters:max_event_id"] = persisted_max
    corrupt.state.flush_cache()
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "persist_event_counter", lambda *args: None)
        corrupt.close(update_status=False)
    reopened = SessionStore(b.path)
    try:
        _, eid = append(reopened, text="next")
        assert eid > max(3, persisted_max)
        before = reopened.submit(capture_events, reopened).result()
        assert copy_events_into(a, reopened) == 0
        assert reopened.submit(capture_events, reopened).result() == before
        assert len(reopened.submit(reopened.get_events, "alice").result()) == 2
    finally:
        reopened.close(update_status=False)
