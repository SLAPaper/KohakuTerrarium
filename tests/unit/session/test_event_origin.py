"""Persistent event provenance survives retries, hops, reopen, and forks."""

from contextlib import closing
from copy import deepcopy
from uuid import uuid4

import pytest
from kohakuvault import KVault

import kohakuterrarium.session.store as store_module
from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.event_origin import (
    EVENT_ORIGIN,
    STORE_UUID,
    LocalOriginIndex,
    append_mirrored,
    capture_events,
    import_events,
    store_uuid,
)
from kohakuterrarium.session.feedback_admission import persist_feedback
from kohakuterrarium.session.native_ack import persist_ack
from kohakuterrarium.session.output_delivery import (
    persist_output_delivery,
    validate_output_source,
)
from kohakuterrarium.session.text_recovery import prove_generation_origin
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


def _legacy_collision(store):
    for name in ("legacy_parent", "legacy_child"):
        store.submit(
            store.append_event,
            name,
            "user_message",
            {"content": name, "event_id": 1, "ts": 10},
        ).result()


def test_cross_namespace_capture_import_retry_and_new_global_cursor(stores):
    source, target, _ = stores
    _legacy_collision(source)
    _legacy_collision(target)
    uid, captured = source.submit(capture_events, source).result()
    assert {row[3] for row in captured} == {
        (uid, "legacy_parent", 1),
        (uid, "legacy_child", 1),
    }
    first = target.submit(import_events, target, (uid, captured)).result()
    assert all(receipt["inserted"] for receipt in first)
    assert {receipt["destination"][3] for receipt in first} == {2, 3}
    before = target.submit(capture_events, target).result()
    repeat = target.submit(import_events, target, (uid, captured)).result()
    assert not any(receipt["inserted"] for receipt in repeat)
    assert [r["destination"] for r in repeat] == [r["destination"] for r in first]
    assert target.submit(capture_events, target).result() == before
    assert target.submit(target.max_event_id, "legacy_parent").result() == 3
    assert [
        r["event_id"]
        for r in target.submit(target.get_events, "legacy_parent", 1).result()
    ] == [3]
    _, new_id = append(target, text="after import")
    assert new_id == 4


@pytest.mark.parametrize("duplicate_origin", [False, True])
def test_capture_rejects_same_namespace_id_even_with_distinct_origins(
    stores, duplicate_origin
):
    source, destination, _ = stores
    append(source)
    uid = source.submit(store_uuid, source).result()
    origin = uid if duplicate_origin else str(uuid4())
    source.submit(
        lambda: source.events.__setitem__(
            "alice:e000001",
            {
                "type": "user_message",
                "event_id": 1,
                "content": "corrupt duplicate",
                EVENT_ORIGIN: [origin, "alice", 1],
            },
        )
    ).result()
    for operation in (
        lambda: source.submit(capture_events, source).result(),
        lambda: copy_events_into(source, destination),
    ):
        with pytest.raises(ValueError, match="Duplicate stored"):
            operation()
    assert destination.submit(destination.get_events, "alice").result() == []


def test_resume_generation_survives_legacy_same_namespace_event_id_collision(stores):
    store, _, _ = stores
    store.submit(
        store.append_event,
        "alice",
        "plugin_hook_timing",
        {
            "hook": "on_agent_stop",
            "plugin": "context_files",
            "event_id": 78283,
            "ts": 10,
        },
        turn_index=149,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    store.submit(
        store.append_event,
        "alice",
        "user_input",
        {
            "content": [{"text": "可以继续了", "type": "text"}],
            "event_id": 78283,
            "ts": 20,
        },
        turn_index=150,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    store.submit(store.flush).result()
    before = store.submit(store.get_events, "alice").result()
    assert [row["event_id"] for row in before] == [78283, 78283]
    assert [row["type"] for row in before] == ["plugin_hook_timing", "user_input"]

    generation = {
        "schema": 1,
        "generation_id": "generation-resume",
        "source": {"model": "scripted"},
    }
    _, event_id = store.submit(
        store.append_event,
        "alice",
        "assistant_generation_start",
        deepcopy(generation),
        turn_index=150,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    store.submit(store.flush).result()
    proof = store.submit(
        prove_generation_origin,
        store,
        "alice",
        generation,
        150,
        1,
        [],
        event_id=event_id,
    ).result()
    assert proof["generation_id"] == "generation-resume"
    assert proof["event_id"] == event_id
    assert event_id != 78283
    receipt = store.submit(
        persist_output_delivery,
        store,
        "alice",
        generation,
        {
            "output_id": "one",
            "target": "notice",
            "content": "body",
            "success": True,
            "error": None,
            "completed_at": "2026-10-04",
            "feedback_line": "notice: body",
        },
    ).result()
    assert receipt["payload"]["generation_id"] == "generation-resume"
    events = store.submit(store.get_events, "alice").result()
    assert [
        {key: row[key] for key in ("type", "event_id") if key in row}
        for row in events[:2]
    ] == [
        {"type": "plugin_hook_timing", "event_id": 78283},
        {"type": "user_input", "event_id": 78283},
    ]
    assert events[-1]["type"] == "output_delivery"


@pytest.mark.parametrize("duplicate_coordinate", [False, True])
def test_local_index_never_resolves_last_writer_for_ambiguous_reference(
    stores, duplicate_coordinate
):
    store, _, _ = stores
    append(store)
    uid = store.submit(store_uuid, store).result()
    collision = uid if not duplicate_coordinate else str(uuid4())
    store.submit(
        lambda: store.events.__setitem__(
            "alice:e000001",
            {
                "type": "user_message",
                "event_id": 1,
                "content": "second",
                EVENT_ORIGIN: [collision, "alice", 1],
            },
        )
    ).result()
    store.submit(
        store.append_event, "bob", "user_message", {"content": "unrelated"}
    ).result()
    _, captured = store.submit(capture_events, store, unique=False).result()
    index = LocalOriginIndex(captured, "alice")
    for ref in {(uid, "alice", 1), (collision, "alice", 1)}:
        with pytest.raises(HistoryCompatibilityError, match="ambiguous"):
            index.get(ref)
        with pytest.raises(HistoryCompatibilityError, match="ambiguous"):
            index[ref]
    for read_rows in (
        lambda: list(index.values()),
        lambda: list(index.items()),
        lambda: dict(index),
    ):
        with pytest.raises(HistoryCompatibilityError, match="ambiguous"):
            read_rows()
    with pytest.raises(TypeError):
        index[(uid, "alice", 1)] = {"content": "bypass"}
    bob = LocalOriginIndex(captured, "bob")
    assert bob[(uid, "bob", 2)]["content"] == "unrelated"
    assert index.get((uid, "alice", 100), "missing") == "missing"


@pytest.mark.parametrize("generation_first", [False, True])
def test_local_generation_proof_rejects_its_own_colliding_origin(
    stores, generation_first
):
    store, _, _ = stores
    generation = {
        "schema": 1,
        "generation_id": "ambiguous-generation",
        "source": {"model": "scripted"},
    }
    records = [
        ("plugin_hook_timing", {"hook": "on_agent_stop"}),
        ("assistant_generation_start", generation),
    ]
    if generation_first:
        records.reverse()
    for kind, data in records:
        store.submit(
            store.append_event,
            "alice",
            kind,
            {**deepcopy(data), "event_id": 7},
            turn_index=1,
            branch_id=1,
            parent_branch_path=[],
        ).result()
    before = store.submit(store.get_events, "alice").result()
    with pytest.raises(HistoryCompatibilityError, match="ambiguous"):
        store.submit(
            prove_generation_origin, store, "alice", generation, 1, 1, []
        ).result()
    assert [
        {k: v for k, v in row.items() if k != "_kt_event_coordinate_v1"}
        for row in store.submit(store.get_events, "alice").result()
    ] == before


@pytest.mark.parametrize("activated", [False, True])
def test_mirror_does_not_introduce_new_cross_namespace_id_collision(stores, activated):
    store, _, _ = stores
    store.submit(
        store.append_event, "alice", "message", {"event_id": 4, "content": "parent"}
    ).result()
    if activated:
        store.submit(store_uuid, store).result()
    before = store.submit(store.get_all_events).result()
    with pytest.raises(ValueError, match="Conflicting mirrored"):
        store.submit(
            append_mirrored,
            store,
            "other",
            "message",
            {"event_id": 4, "content": "new collision"},
        ).result()
    assert store.submit(store.get_all_events).result() == before


def test_mirror_retry_in_legacy_collision_selects_its_own_namespace(stores):
    store, _, _ = stores
    _legacy_collision(store)
    before = store.submit(store.get_all_events).result()
    for name in ("legacy_parent", "legacy_child"):
        row = deepcopy(store.submit(store.get_events, name).result()[0])
        result = store.submit(
            append_mirrored, store, name, row.pop("type"), row
        ).result()
        assert result[1] == 1
    assert store.submit(store.get_all_events).result() == before
    with pytest.raises(ValueError, match="Conflicting mirrored"):
        store.submit(
            append_mirrored,
            store,
            "legacy_parent",
            "user_message",
            {"event_id": 1, "content": "changed", "ts": 10},
        ).result()
    assert store.submit(store.get_all_events).result() == before


def test_legacy_collision_does_not_block_child_feedback_or_native_ack(stores):
    store, _, _ = stores
    _legacy_collision(store)
    origin = {
        "runtime_scope_id": "scope",
        "delivery_id": "child-delivery",
        "execution_class": "direct",
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
        "generation_id": "generation",
    }
    declaration = {
        "generation_id": "generation",
        "veto_group_id": "scope:generation",
        "tool_calls": [
            {"id": "native-child", "function": {"name": "child", "arguments": "{}"}}
        ],
    }
    _, declaration_id = store.submit(
        store.append_event,
        "alice",
        "assistant_tool_calls",
        declaration,
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    dispatch = {"job_id": "child", "origin": origin, "tool_call_id": "native-child"}
    store.submit(
        store.append_event,
        "alice",
        "subagent_call",
        dispatch,
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    uid = store.submit(store_uuid, store).result()
    ack = {
        "job_id": "child",
        "origin": origin,
        "tool_call_id": "native-child",
        "phase": "ack",
        "declaration_origin": [uid, "alice", declaration_id],
    }
    first = store.submit(persist_ack, store, "alice", ack).result()
    before = store.submit(store.get_events, "alice").result()
    repeated = store.submit(persist_ack, store, "alice", ack).result()
    assert repeated[0] == first[0]
    assert repeated[1]["dispatch_origin"] == first[1]["dispatch_origin"]
    assert repeated[1]["declaration_origin"] == first[1]["declaration_origin"]
    assert store.submit(store.get_events, "alice").result() == before
    with pytest.raises(HistoryCompatibilityError, match="conflicts"):
        store.submit(
            persist_ack, store, "alice", {**ack, "content": "changed retry"}
        ).result()
    direct = {
        "job_id": "direct-child",
        "origin": {**origin, "delivery_id": "direct-return"},
    }
    for kind in ("subagent_call", "subagent_result"):
        store.submit(
            store.append_event,
            "alice",
            kind,
            deepcopy(direct),
            turn_index=1,
            branch_id=1,
            parent_branch_path=[],
        ).result()
    proof = {**direct, "mode": "direct_child"}
    receipt = store.submit(
        persist_feedback, store, "alice", "child completed", [proof], 1, 1, []
    ).result()
    row = store.submit(store.get_events, "alice").result()[-1]
    assert row["event_id"] == receipt[2] and row["type"] == "tool_feedback_admitted"
    assert len(row["source_event_origins"]) == 2
    assert {tuple(ref) for ref in row["source_event_origins"]} <= {
        ref
        for name, _, _, ref in store.submit(capture_events, store).result()[1]
        if name == "alice"
    }
    before = store.submit(store.get_events, "alice").result()
    with pytest.raises(HistoryCompatibilityError, match="missing or ambiguous"):
        store.submit(
            persist_feedback,
            store,
            "alice",
            "foreign",
            [{**proof, "origin": {**proof["origin"], "runtime_scope_id": "foreign"}}],
            1,
            1,
            [],
        ).result()
    assert store.submit(store.get_events, "alice").result() == before


def test_legacy_local_ack_and_feedback_reject_colliding_referenced_receipts(stores):
    store, _, _ = stores
    origin = {
        "runtime_scope_id": "scope",
        "delivery_id": "dispatch",
        "execution_class": "background",
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
        "generation_id": "generation",
    }
    uid = store.submit(store_uuid, store).result()
    declaration = {
        "generation_id": "generation",
        "veto_group_id": "scope:generation",
        "tool_calls": [
            {"id": "native-child", "function": {"name": "child", "arguments": "{}"}}
        ],
    }
    _, declaration_id = store.submit(
        store.append_event,
        "alice",
        "assistant_tool_calls",
        declaration,
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    dispatch = {"job_id": "child", "origin": origin, "tool_call_id": "native-child"}
    store.submit(
        store.append_event,
        "alice",
        "subagent_call",
        deepcopy(dispatch),
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    ack = {
        **dispatch,
        "phase": "ack",
        "declaration_origin": [uid, "alice", declaration_id],
    }
    receipt, _ = store.submit(persist_ack, store, "alice", ack).result()
    store.submit(
        lambda: store.events.__setitem__(
            "alice:e900000",
            {
                "event_id": receipt[2],
                "type": "plugin_hook_timing",
                "hook": "legacy collision",
            },
        )
    ).result()
    before = store.submit(store.get_events, "alice").result()
    with pytest.raises(HistoryCompatibilityError, match="ambiguous"):
        store.submit(persist_ack, store, "alice", ack).result()
    assert store.submit(store.get_events, "alice").result() == before
    plain_origin = {**origin, "delivery_id": "plain-dispatch"}
    _, plain_id = store.submit(
        store.append_event,
        "alice",
        "tool_call",
        {"job_id": "plain", "origin": plain_origin},
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    store.submit(
        lambda: store.events.__setitem__(
            "alice:e900001",
            {
                "event_id": plain_id,
                "type": "plugin_hook_timing",
                "hook": "another collision",
            },
        )
    ).result()
    proof = {"job_id": "plain", "origin": plain_origin, "mode": "dispatch_ack"}
    before = store.submit(store.get_events, "alice").result()
    with pytest.raises(HistoryCompatibilityError, match="ambiguous"):
        store.submit(
            persist_feedback, store, "alice", "accepted dispatch", [proof], 1, 1, []
        ).result()
    assert store.submit(store.get_events, "alice").result() == before
    assert "dispatch_origin" not in proof


def test_legacy_collision_preserves_output_generation_receipt(stores):
    store, _, _ = stores
    _legacy_collision(store)
    generation = {
        "schema": 1,
        "generation_id": "generation-one",
        "source": {"model": "scripted"},
    }
    store.submit(
        store.append_event,
        "alice",
        "assistant_generation_start",
        deepcopy(generation),
        turn_index=1,
        branch_id=1,
        parent_branch_path=[],
    ).result()
    payload = {
        "output_id": "one",
        "target": "notice",
        "content": "body",
        "success": True,
        "error": None,
        "completed_at": "2026-10-03",
        "feedback_line": "notice: body",
    }
    proof = store.submit(
        persist_output_delivery, store, "alice", generation, payload
    ).result()
    captured = store.submit(capture_events, store).result()[1]
    index = {ref: row for name, _, row, ref in captured if name == "alice"}
    assert validate_output_source(proof, index) == tuple(proof["event_origin"])
    receipt = store.submit(
        persist_feedback, store, "alice", "notice: body", [proof], 1, 1, []
    ).result()
    assert store.submit(store.get_events, "alice").result()[-1][
        "source_event_origins"
    ] == [proof["event_origin"]]
    assert receipt[2] > proof["event_origin"][2]


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


@pytest.mark.parametrize("persisted_max", [0, 1, 100])
def test_legacy_reopen_reconciles_counter_before_first_append(tmp_path, persisted_max):
    path = tmp_path / "legacy.kohakutr"
    with closing(SessionStore(path)) as original:
        for namespace in ("alice", "alice:child", "alice"):
            original.submit(
                original.append_event, namespace, "user_message", {"content": namespace}
            ).result()
        before = original.submit(original.get_events, "alice").result()
        assert original.meta.get(STORE_UUID) is None
    with closing(KVault(str(path), table="state")) as native:
        native.enable_auto_pack()
        native["counters:max_event_id"] = persisted_max
    with closing(SessionStore(path)) as resumed:
        _, event_id = resumed.submit(
            resumed.append_event, "alice", "assistant_generation_start", {}
        ).result()
        assert event_id == max(3, persisted_max) + 1
        assert resumed.submit(resumed.get_events, "alice").result()[:-1] == before
        _, captured = resumed.submit(capture_events, resumed).result()
        assert len(captured) == 4
        assert len({(name, row["event_id"]) for name, _, row, _ in captured}) == 4
    with closing(SessionStore(path)) as again:
        assert (
            again.submit(
                again.append_event,
                "alice",
                "user_message",
                {"content": "second reopen"},
            ).result()[1]
            > event_id
        )


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
