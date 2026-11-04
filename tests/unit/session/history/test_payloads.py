"""Temporary restore descriptors retain metadata and one borrowed payload."""

import os
import sqlite3
from contextlib import closing
from weakref import ref

import pytest
from kohakuvault import KVault

from kohakuterrarium.core.agent.history.message_history import replay_with_settled_tools
from kohakuterrarium.session.history.payloads import (
    PAGE_BYTES,
    PAGE_ROWS,
    event_pages,
    read_descriptors,
)
from kohakuterrarium.session.history.read import materialize_event_paths
from kohakuterrarium.session.history.replay import dedupe_adjacent_duplicate_events
from kohakuterrarium.session.store import SessionStore


def test_descriptors_borrow_one_body_and_preserve_public_values(tmp_path, monkeypatch):
    with closing(SessionStore(tmp_path / "descriptors.kohakutr")) as store:
        for index in range(193):
            store.append_event(
                "a",
                "user_message",
                {
                    "content": str(index) * 2000,
                    "nested": {"x": [index]},
                    "none": None,
                    "flag": True,
                },
                turn_index=index + 1,
                branch_id=1,
                parent_branch_path=[],
            )
        store.append_event("a:e-other", "user_message", {"content": "neighbor"})
        store.flush()
        calls = []
        original = KVault.get_many

        def observed(table, keys, *args, **kwargs):
            if table is store.events:
                calls.append(tuple(keys))
            return original(table, keys, *args, **kwargs)

        monkeypatch.setattr(KVault, "get_many", observed)
        events = read_descriptors(store, "a")
        assert sum(map(len, calls)) == 193
        assert max(map(len, calls)) <= PAGE_ROWS
        reader = events[0]._reader
        assert reader._values == {}
        assert all(
            "content" not in event._header and "nested" not in event._header
            for event in events
        )
        count = len(calls)
        assert [event["turn_index"] for event in events] == list(range(1, 194))
        assert len(calls) == count
        assert len(dedupe_adjacent_duplicate_events(events)) == 193
        assert len(calls) == count
        assert events[0]["content"] == "0" * 2000
        assert calls[count:] == [(events[0]._key,)]
        assert events[0]["nested"] == {"x": [0]}
        assert len(calls) == count + 1
        assert events[1]["nested"] == {"x": [1]}
        assert calls[-1] == (events[1]._key,)
        assert len(reader._values) == 1
        assert events[-1]["nested"] == {"x": [192]}
        assert calls[-1] == (events[-1]._key,)
        assert len(reader._values) == 1
        assert events[0]["content"] == "0" * 2000
        assert calls[-1] == (events[0]._key,)
        assert all(len(keys) == 1 for keys in calls[count:])
        with pytest.raises(TypeError):
            events[0]["content"] = "changed"
        with pytest.raises(KeyError):
            events[0]["absent"]
        assert materialize_event_paths(events) == store.get_events("a")
        reader.release()
        assert reader._values == {}


def test_pages_bound_encoded_bytes_and_allow_one_oversized_value(tmp_path):
    with closing(SessionStore(tmp_path / "pages.kohakutr")) as store:
        for size in (400000, 400000, 400000, 2 * PAGE_BYTES):
            store.append_event("a", "activity", {"body": os.urandom(size)})
        store.flush()
        pages = list(event_pages(store, "a"))
        assert sum(map(len, pages)) == 4
        with closing(sqlite3.connect(store.path)) as connection:
            for page in pages:
                size = sum(
                    connection.execute(
                        "SELECT length(value) FROM events WHERE key=?", (key,)
                    ).fetchone()[0]
                    for key in page
                )
                assert size <= PAGE_BYTES or len(page) == 1


def test_previous_oversized_body_is_released_before_decoding_next_page(
    tmp_path, monkeypatch
):
    class ObservedEvent(dict):
        pass

    with closing(SessionStore(tmp_path / "release.kohakutr")) as store:
        for _ in range(2):
            store.append_event("a", "activity", {"body": os.urandom(PAGE_BYTES + 100)})
        store.flush()
        previous, overlaps = [], []
        original = KVault.get_many

        def observed(table, keys, *args, **kwargs):
            if table is store.events:
                overlaps.append(any(body() is not None for body in previous))
                values = [
                    ObservedEvent(value)
                    for value in original(table, keys, *args, **kwargs)
                ]
                previous.extend(ref(value) for value in values)
                return values
            return original(table, keys, *args, **kwargs)

        monkeypatch.setattr(KVault, "get_many", observed)
        events = read_descriptors(store, "a")
        assert len(events) == 2
        assert len(events[0]["body"]) == PAGE_BYTES + 100
        assert len(events[1]["body"]) == PAGE_BYTES + 100
        assert not any(overlaps)


def test_missing_lazy_payload_releases_the_previous_body(tmp_path):
    with closing(SessionStore(tmp_path / "missing.kohakutr")) as store:
        for index in range(2):
            store.append_event("a", "activity", {"body": [index]})
        events = read_descriptors(store, "a")
        assert events[0]["body"] == [0]
        del store.events[events[1]._key]
        with pytest.raises(KeyError):
            events[1]["body"]
        assert events[0]._reader._values == {}
        assert events[0]["body"] == [0]


def test_new_operation_observes_direct_kv_edits_and_replays_identically(tmp_path):
    with closing(SessionStore(tmp_path / "changed.kohakutr")) as store:
        key, _ = store.append_event(
            "a", "user_message", {"content": "old"}, turn_index=1, branch_id=1
        )
        store.append_event(
            "a",
            "tool_call",
            {"call_id": "c", "name": "read", "args": {"path": "x"}},
            turn_index=1,
            branch_id=1,
        )
        first = read_descriptors(store, "a")
        assert first[0]["content"] == "old"
        event = store.events[key]
        event["content"] = "updated" * 1000
        store.events[key] = event
        current = read_descriptors(store, "a")
        assert current[0]["content"] == "updated" * 1000
        assert replay_with_settled_tools(
            current, resumable=True
        ) == replay_with_settled_tools(store.get_events("a"), resumable=True)


def test_signature_collisions_compare_full_values_and_preserve_numeric_equality(
    tmp_path,
):
    with closing(SessionStore(tmp_path / "collisions.kohakutr")) as store:
        for value in (1, True, 2, 3):
            store.append_event("a", "activity", {"value": value, "body": "x" * 10000})
        events = read_descriptors(store, "a")
        assert events[0]._signature_hash == events[1]._signature_hash
        for event in events:
            event._signature_hash = 0
        assert [
            event["value"] for event in dedupe_adjacent_duplicate_events(events)
        ] == [1, 2, 3]
