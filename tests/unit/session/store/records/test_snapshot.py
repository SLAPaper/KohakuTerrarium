from pathlib import Path

import pytest

from kohakuterrarium.session.store.records.snapshot import (
    PAYLOAD_MARKER,
    load_headers,
    load_branch_snapshot,
    load_records,
)
from kohakuterrarium.session.store import SessionStore


@pytest.fixture
def saved(tmp_path):
    store = SessionStore(tmp_path / "checkpoints.kohakutr")
    try:
        yield store
    finally:
        store.close(update_status=False)


def _save(store, branch, content, *, agent="alice"):
    store.save_branch_snapshot(
        agent,
        [
            {
                "role": "user",
                "content": content,
                "metadata": {"turn_index": 1, "branch_id": branch},
            }
        ],
        branch={"turn_index": 1, "branch_id": branch, "parent_branch_path": []},
        watermark=branch,
    )


def _payload_keys(store, agent="alice"):
    return [
        key.decode() if isinstance(key, bytes) else key
        for key in store.state.keys(prefix=f"{agent}{PAYLOAD_MARKER}")
    ]


def test_headers_exclude_messages_and_selected_read_loads_one_payload(
    saved, monkeypatch
):
    for branch in range(1, 5):
        _save(saved, branch, f"branch-{branch}")
    read = type(saved.state).get
    reads = []

    def observe(vault, key, *args, **kwargs):
        if vault is saved.state:
            reads.append(key)
        return read(vault, key, *args, **kwargs)

    monkeypatch.setattr(type(saved.state), "get", observe)
    headers = load_headers(saved, "alice")
    assert len(headers) == 4
    assert all("messages" not in header for header in headers)
    assert reads == ["alice:branch_snapshots"]
    record = load_branch_snapshot(saved, "alice", {1: 2})
    assert record["messages"][0]["content"] == "branch-2"
    assert [key for key in reads if PAYLOAD_MARKER in key] == [
        headers[1]["messages_key"]
    ]
    record["messages"][0]["metadata"]["branch_id"] = 99
    assert (
        load_branch_snapshot(saved, "alice", {1: 2})["messages"][0]["metadata"][
            "branch_id"
        ]
        == 2
    )


def test_legacy_records_read_without_mutation_and_split_on_next_save(saved):
    old = [
        {
            "path": [[1, 1]],
            "watermark": 1,
            "messages": [{"role": "user", "content": "legacy"}],
            "compact_state": {"compact_count": 0, "last_compact_time": 0.0},
        }
    ]
    saved.state["alice:branch_snapshots"] = old
    assert load_branch_snapshot(saved, "alice", {1: 1}) == old[0]
    assert saved.state["alice:branch_snapshots"] == old
    assert _payload_keys(saved) == []
    _save(saved, 2, "new branch")
    assert all(
        "messages_key" in header and "messages" not in header
        for header in saved.state["alice:branch_snapshots"]
    )
    assert (
        load_branch_snapshot(saved, "alice", {1: 1})["messages"][0]["content"]
        == "legacy"
    )
    assert (
        load_branch_snapshot(saved, "alice", {1: 2})["messages"][0]["content"]
        == "new branch"
    )


@pytest.mark.parametrize("kind", ["missing", "malformed", "foreign", "nested"])
def test_bad_best_payload_falls_back_to_valid_checkpoint(saved, kind):
    _save(saved, 1, "valid")
    header = load_headers(saved, "alice")[0]
    ref = f"{'bob' if kind == 'foreign' else 'alice'}{PAYLOAD_MARKER}{'f' * 32}"
    if kind == "nested":
        ref = f"alice{PAYLOAD_MARKER}nested{PAYLOAD_MARKER}{'f' * 32}"
    bad = {**header, "watermark": 2, "messages_key": ref}
    if kind != "missing":
        saved.state[ref] = [
            {"role": "bad" if kind == "malformed" else "user", "content": "invalid"}
        ]
    saved.state["alice:branch_snapshots"] = [header, bad]
    assert (
        load_branch_snapshot(saved, "alice", {1: 1})["messages"][0]["content"]
        == "valid"
    )


def test_replacement_and_clear_reclaim_only_own_payloads(saved):
    _save(saved, 1, "original")
    old = _payload_keys(saved)[0]
    _save(saved, 1, "replacement")
    assert old not in _payload_keys(saved)
    assert len(_payload_keys(saved)) == 1
    _save(saved, 1, "other", agent="bob")
    saved.state["alice:unrelated"] = "keep"
    saved.save_conversation("alice", [{"role": "user", "content": "latest"}])
    assert _payload_keys(saved) == []
    assert len(_payload_keys(saved, "bob")) == 1
    assert saved.state["alice:unrelated"] == "keep"
    assert load_records(saved, "alice") == []


def test_cleanup_preserves_nested_agent_namespace(saved):
    nested = f"alice{PAYLOAD_MARKER}nested"
    _save(saved, 1, "other namespace", agent=nested)
    _save(saved, 1, "parent namespace")
    saved.save_conversation("alice", [{"role": "user", "content": "cleared"}])
    assert (
        load_branch_snapshot(saved, nested, {1: 1})["messages"][0]["content"]
        == "other namespace"
    )


def test_failed_batch_does_not_publish_dangling_references(saved, monkeypatch):
    _save(saved, 1, "original")
    before = saved.state["alice:branch_snapshots"]
    keys = _payload_keys(saved)
    put = type(saved.state).put_many

    def fail(vault, rows):
        if vault is saved.state:
            raise OSError("write failed")
        return put(vault, rows)

    monkeypatch.setattr(type(saved.state), "put_many", fail)
    with pytest.raises(OSError, match="write failed"):
        _save(saved, 2, "never committed")
    assert saved.state["alice:branch_snapshots"] == before
    assert _payload_keys(saved) == keys
    assert (
        load_branch_snapshot(saved, "alice", {1: 1})["messages"][0]["content"]
        == "original"
    )


def test_cleanup_failure_preserves_published_checkpoint_and_retries(saved, monkeypatch):
    _save(saved, 1, "old")
    old = _payload_keys(saved)[0]
    pop = type(saved.state).pop

    def fail(vault, key, *args):
        if vault is saved.state and key == old:
            raise OSError("cleanup failed")
        return pop(vault, key, *args)

    with monkeypatch.context() as patch:
        patch.setattr(type(saved.state), "pop", fail)
        _save(saved, 1, "published")
        assert (
            load_branch_snapshot(saved, "alice", {1: 1})["messages"][0]["content"]
            == "published"
        )
        assert old in _payload_keys(saved)
    _save(saved, 1, "next")
    assert old not in _payload_keys(saved)


def test_save_load_branch_snapshot_lifecycle(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path / "s.kohakutr"))
    msgs_a = [
        {
            "role": "user",
            "content": "u1",
            "metadata": {"turn_index": 1, "branch_id": 1},
        },
        {"role": "assistant", "content": "a1"},
        {
            "role": "user",
            "content": "u2",
            "metadata": {"turn_index": 2, "branch_id": 1},
        },
    ]
    store.save_branch_snapshot(
        "alice",
        msgs_a,
        branch={"turn_index": 2, "branch_id": 1, "parent_branch_path": [[1, 1]]},
        watermark=20,
        compact_state={"compact_count": 1, "last_compact_time": 100.0},
    )
    msgs_b = [
        {
            "role": "user",
            "content": "u1",
            "metadata": {"turn_index": 1, "branch_id": 1},
        },
        {
            "role": "user",
            "content": "u2_b",
            "metadata": {"turn_index": 2, "branch_id": 2},
        },
    ]
    store.save_branch_snapshot(
        "alice",
        msgs_b,
        branch={"turn_index": 2, "branch_id": 2, "parent_branch_path": [[1, 1]]},
        watermark=25,
        compact_state={"compact_count": 1, "last_compact_time": 100.0},
    )

    records = load_records(store, "alice")
    assert [r["path"] for r in records] == [[[1, 1], [2, 1]], [[1, 1], [2, 2]]]

    records[0]["messages"][0]["content"] = "mutated"
    assert load_records(store, "alice")[0]["messages"][0]["content"] == "u1"

    snap_a = load_branch_snapshot(store, "alice", {1: 1, 2: 1})
    assert snap_a is not None and len(snap_a["messages"]) == 3
    assert snap_a["path"] == [[1, 1], [2, 1]]

    b_a3 = {"turn_index": 3, "branch_id": 1, "parent_branch_path": [[1, 1], [2, 1]]}
    store.save_branch_snapshot("alice", msgs_a, branch=b_a3, watermark=30)
    recs3 = load_records(store, "alice")
    assert [r["path"] for r in recs3] == [[[1, 1], [2, 2]], [[1, 1], [2, 1], [3, 1]]]

    store.save_conversation("alice", [{"role": "user", "content": "legacy"}])
    assert load_records(store, "alice") == []
    store.close()
