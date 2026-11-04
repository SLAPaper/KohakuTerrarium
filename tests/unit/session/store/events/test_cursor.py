"""Native event seeks stay bounded and exact to the requested namespace."""

from contextlib import closing

import pytest

from kohakuvault import KVault

from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.store.events.cursor import event_rows


def test_reverse_turn_seek_reads_one_page_and_skips_neighbor_namespace(
    tmp_path, monkeypatch
):
    with closing(SessionStore(tmp_path / "tail.kohakutr")) as store:
        for i in range(200):
            store.append_event(
                "a", "user_message", {"content": str(i)}, turn_index=i + 1
            )
        store.append_event("a", "processing_end", {})
        store.append_event(
            "a:e-other", "user_message", {"content": "neighbor"}, turn_index=900
        )
        store.flush()
        reads = []
        original = KVault.get_many

        def observed(table, keys, *args, **kwargs):
            if table is store.events:
                reads.extend(keys)
            return original(table, keys, *args, **kwargs)

        monkeypatch.setattr(KVault, "get_many", observed)
        event = store.get_last_turn_event("a")
        assert event["turn_index"] == 200
        assert len(reads) <= 64
        with closing(event_rows(store, "a")) as rows:
            assert len(list(rows)) == 201


def test_empty_turn_seek_and_unflushed_rows(tmp_path):
    with closing(SessionStore(tmp_path / "empty.kohakutr")) as store:
        assert store.get_last_turn_event("a") is None
        store.append_event("a", "processing_end", {})
        assert store.get_last_turn_event("a") is None
        store.append_event("a", "user_message", {"content": "new"}, turn_index=1)
        assert store.get_last_turn_event("a")["content"] == "new"


@pytest.mark.parametrize("suffix", ["z", "①"])
def test_reverse_seek_keeps_opaque_legacy_keys_in_the_exact_namespace(tmp_path, suffix):
    with closing(SessionStore(tmp_path / "legacy.kohakutr")) as store:
        store.append_event("a", "user_message", {}, turn_index=1, branch_id=1)
        store.events[f"a:e{suffix}"] = {
            "type": "processing_start",
            "event_id": 2,
            "turn_index": 2,
            "branch_id": 3,
            "parent_branch_path": [[1, 1]],
        }
        store.append_event(f"a:e{suffix}", "user_message", {}, turn_index=900)
        expected = next(
            e for e in reversed(store.get_events("a")) if e.get("turn_index")
        )
        assert store.get_last_turn_event("a") == expected
        with closing(event_rows(store, "a")) as rows:
            assert [event for _, event in rows] == store.get_events("a")
