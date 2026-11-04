"""Normal checkpoints save snapshots without authenticating historical events."""

from contextlib import closing

from kohakuterrarium.session.store import SessionStore
from tests.helpers.event_payload_reads import track_event_payload_reads


def test_checkpoint_does_not_read_events_or_write_derived_context(
    tmp_path, monkeypatch
):
    path = tmp_path / "checkpoint.kohakutr"
    with closing(SessionStore(path)) as store:
        store.append_event(
            "a", "user_message", {"content": "question"}, turn_index=1, branch_id=1
        )
    reads = track_event_payload_reads(monkeypatch, path)
    with closing(SessionStore(path)) as store:
        reads.clear()
        messages = [
            {
                "role": "user",
                "content": "question",
                "metadata": {"turn_index": 1, "branch_id": 1},
            },
            {"role": "assistant", "content": "answer"},
        ]
        store.save_branch_snapshot(
            "a",
            messages,
            branch={"turn_index": 1, "branch_id": 1, "parent_branch_path": []},
            watermark=1,
        )
        assert reads == []
        assert store.load_conversation("a") == messages
        assert store.load_branch_snapshot("a", {1: 1})["messages"] == messages
        assert not any(
            b"recovery_" in key or b"history_projection" in key
            for key in store.state.keys()
        )
