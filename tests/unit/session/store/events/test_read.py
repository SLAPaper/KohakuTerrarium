"""Physical sequence fences do not change public event reads."""

from contextlib import closing

import pytest

from kohakuterrarium.session.store import SessionStore


@pytest.mark.parametrize("sequence", [0, 999999])
def test_sequence_fence_keeps_buffered_rows_and_excludes_later_mirrored_ids(
    tmp_path, sequence
):
    with closing(SessionStore(tmp_path / "fenced.kohakutr")) as store:
        store._event_seq["a"] = sequence
        store.append_event("a", "text", {"content": "before", "event_id": 10})
        fence = store.event_sequence_boundary("a")
        assert fence == sequence + 1
        store.append_event("a", "text", {"content": "after", "event_id": 5})
        store.append_event("b", "text", {"content": "other"})
        assert [
            e["content"] for e in store._iter_events("a", before_sequence=fence)
        ] == ["before"]
        assert {e["content"] for e in store.get_events("a")} == {"before", "after"}
        assert store.event_sequence_boundary("a") == sequence + 2
        assert store.event_sequence_boundary("empty") == 0


def test_opaque_legacy_keys_remain_readable(tmp_path):
    with closing(SessionStore(tmp_path / "invalid-key.kohakutr")) as store:
        store.append_event("a", "text", {"content": "own", "event_id": 10})
        store.events["a:e①"] = {"type": "text", "event_id": 5, "content": "invalid"}
        assert [event["content"] for event in store.get_events("a")] == [
            "own",
            "invalid",
        ]
        assert [
            event["content"] for event in store._iter_events("a", before_sequence=1)
        ] == ["own", "invalid"]


def test_overlapping_namespaces_do_not_share_event_reads(tmp_path):
    with closing(SessionStore(tmp_path / "namespaces.kohakutr")) as store:
        store.append_event("a:e-other", "text", {"content": "other", "event_id": 9})
        store.append_event("a", "text", {"content": "own", "event_id": 10})
        assert [event["content"] for event in store.get_events("a")] == ["own"]
        assert [event["content"] for event in store.get_events("a:e-other")] == [
            "other"
        ]
