"""Unit tests for text recovery build and recover functions."""

from copy import deepcopy
import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.event_origin import store_uuid
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.text_recovery import build_record, recover_record


def _create_store(tmp_path, name="session.kohakutr"):
    """Create a SessionStore in the temp directory."""
    path = tmp_path / name
    store = SessionStore(path)
    store.submit(store_uuid, store).result()
    return store


def _seed_generation_declaration(
    store,
    namespace="alice",
    generation_id="gen-1",
    turn_index=1,
    branch_id=1,
    parent_branch_path=None,
    model="scripted-model",
):
    """Seed an assistant_generation_start declaration event."""
    generation = {
        "schema": 1,
        "generation_id": generation_id,
        "source": {"model": model},
    }
    path = parent_branch_path or []
    store.submit(
        store.append_event,
        namespace,
        "assistant_generation_start",
        deepcopy(generation),
        turn_index=turn_index,
        branch_id=branch_id,
        parent_branch_path=deepcopy(path),
    ).result()
    store.submit(store.flush).result()
    return generation


def test_non_user_generation_uses_its_proven_unbranched_coordinates(tmp_path):
    store = _create_store(tmp_path)
    try:
        generation = _seed_generation_declaration(
            store, turn_index=None, branch_id=None
        )
        record = store.submit(
            build_record,
            store,
            "alice",
            "drive-triggered partial",
            generation,
            None,
            None,
            [],
        ).result()
        receipt = store.submit(recover_record, store, "alice", record).result()
        assert receipt["generation_id"] == generation["generation_id"]
        assert receipt.get("turn_index") is None and receipt.get("branch_id") is None
        assert receipt["parent_branch_path"] == []
        with pytest.raises(HistoryCompatibilityError, match="coordinates"):
            store.submit(
                build_record,
                store,
                "alice",
                "wrong partial coordinates",
                generation,
                None,
                1,
                [],
            ).result()
    finally:
        store.close(update_status=False)


def test_build_record_fails_when_declaration_missing(tmp_path):
    """build_record raises HistoryCompatibilityError when no declaration exists."""
    store = _create_store(tmp_path)
    try:
        generation = {
            "schema": 1,
            "generation_id": "gen-missing",
            "source": {"model": "gpt"},
        }
        with pytest.raises(HistoryCompatibilityError, match="declaration"):
            store.submit(
                build_record,
                store,
                "alice",
                "hello partial",
                generation,
                1,
                1,
                [],
            ).result()
    finally:
        store.close()


def test_build_record_fails_when_coords_or_source_mismatch(tmp_path):
    """build_record raises HistoryCompatibilityError when coords do not match declaration."""
    store = _create_store(tmp_path)
    try:
        generation = _seed_generation_declaration(
            store, namespace="alice", turn_index=1, branch_id=1
        )
        # Turn mismatch (requested turn 2, declaration was turn 1)
        with pytest.raises(HistoryCompatibilityError, match="mismatch"):
            store.submit(
                build_record,
                store,
                "alice",
                "hello partial",
                generation,
                2,
                1,
                [],
            ).result()

        # Source mismatch
        conflicting_gen = {
            "schema": 1,
            "generation_id": generation["generation_id"],
            "source": {"model": "different-model"},
        }
        with pytest.raises(HistoryCompatibilityError, match="mismatch"):
            store.submit(
                build_record,
                store,
                "alice",
                "hello partial",
                conflicting_gen,
                1,
                1,
                [],
            ).result()
    finally:
        store.close()


def test_recover_record_rejects_legacy_plain_string(tmp_path):
    """recover_record rejects legacy plain strings without mutating slot or events."""
    store = _create_store(tmp_path)
    try:
        slot_key = "alice:open_text"
        store.submit(
            lambda: store.state.__setitem__(slot_key, "legacy unproven text")
        ).result()
        events_before = store.submit(store.get_events, "alice").result()

        with pytest.raises(HistoryCompatibilityError, match="[Ll]egacy"):
            store.submit(recover_record, store, "alice").result()

        # Verify slot is untouched and no event was appended
        assert (
            store.submit(store.state.get, slot_key).result() == "legacy unproven text"
        )
        assert store.submit(store.get_events, "alice").result() == events_before
    finally:
        store.close()


def test_recover_record_rejects_foreign_store_uuid(tmp_path):
    """recover_record rejects records created for a different store UUID."""
    store_a = _create_store(tmp_path, "store_a.kohakutr")
    store_b = _create_store(tmp_path, "store_b.kohakutr")
    try:
        gen = _seed_generation_declaration(store_a, namespace="alice")
        record = store_a.submit(
            build_record, store_a, "alice", "foreign text", gen, 1, 1, []
        ).result()

        # Attempt to recover record built for store_a in store_b
        with pytest.raises(HistoryCompatibilityError, match="[Ss]tore UUID"):
            store_b.submit(recover_record, store_b, "alice", record).result()
    finally:
        store_a.close()
        store_b.close()


def test_recover_record_rejects_conflicting_payload_for_same_segment(tmp_path):
    """recover_record rejects conflicting payload for an already recovered segment."""
    store = _create_store(tmp_path)
    try:
        gen = _seed_generation_declaration(store, namespace="alice")
        record = store.submit(
            build_record, store, "alice", "original text", gen, 1, 1, []
        ).result()

        # First recovery succeeds
        receipt = store.submit(recover_record, store, "alice", record).result()
        assert receipt["content"] == "original text"

        # Conflicting record with same segment_id but different text
        conflicting_record = deepcopy(record)
        conflicting_record["text"] = "mutated text"

        with pytest.raises(HistoryCompatibilityError, match="[Cc]onflict"):
            store.submit(recover_record, store, "alice", conflicting_record).result()
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["append_receipt", "checkpoint", "clear"])
def test_recovery_retry_reuses_receipt_after_storage_boundary_failure(
    tmp_path, monkeypatch, failure
):
    store = _create_store(tmp_path)
    try:
        generation = _seed_generation_declaration(store)
        record = store.submit(
            build_record, store, "alice", "partial", generation, 1, 1, []
        ).result()
        store.submit(store.state.__setitem__, "alice:open_text", record).result()
        store.submit(store.checkpoint, strict=True).result()
        fired = False
        original_append = SessionStore.append_event
        original_checkpoint = SessionStore.checkpoint
        state_type = type(store.state)
        original_delete = state_type.__delitem__

        def append(candidate, *args, **kwargs):
            nonlocal fired
            result = original_append(candidate, *args, **kwargs)
            if candidate is store and failure == "append_receipt" and not fired:
                fired = True
                raise OSError("recovery receipt lost")
            return result

        def checkpoint(candidate, **kwargs):
            nonlocal fired
            if candidate is store and failure == "checkpoint" and not fired:
                fired = True
                raise OSError("recovery flush failed")
            return original_checkpoint(candidate, **kwargs)

        def delete(table, key):
            nonlocal fired
            if table is store.state and failure == "clear" and not fired:
                fired = True
                raise OSError("recovery clear failed")
            return original_delete(table, key)

        with monkeypatch.context() as patch:
            patch.setattr(SessionStore, "append_event", append)
            patch.setattr(SessionStore, "checkpoint", checkpoint)
            patch.setattr(state_type, "__delitem__", delete)
            with pytest.raises(OSError, match="recovery"):
                store.submit(recover_record, store, "alice").result()
        assert fired
        assert store.submit(store.state.get, "alice:open_text").result() == record
        receipt = store.submit(recover_record, store, "alice").result()
        assert store.submit(recover_record, store, "alice", record).result() == receipt
        rows = store.submit(store.get_events, "alice").result()
        assert (
            len([row for row in rows if row.get("segment_id") == record["segment_id"]])
            == 1
        )
        assert store.submit(store.state.get, "alice:open_text").result() is None
        path = store.path
        store.close(update_status=False)
        reopened = SessionStore(path)
        try:
            assert reopened.submit(recover_record, reopened, "alice").result() is None
            assert (
                reopened.submit(recover_record, reopened, "alice", record).result()
                == receipt
            )
        finally:
            reopened.close(update_status=False)
    finally:
        if not store._closed:
            store.close(update_status=False)


def test_recovery_does_not_clear_a_newer_segment_slot(tmp_path, monkeypatch):
    store = _create_store(tmp_path)
    try:
        generation = _seed_generation_declaration(store)
        old = store.submit(
            build_record, store, "alice", "old", generation, 1, 1, []
        ).result()
        new = store.submit(
            build_record, store, "alice", "new", generation, 1, 1, []
        ).result()
        store.submit(store.state.__setitem__, "alice:open_text", old).result()
        checkpoint = SessionStore.checkpoint
        changed = False

        def replace_slot(candidate, **kwargs):
            nonlocal changed
            if candidate is store and not changed:
                changed = True
                candidate.state["alice:open_text"] = new
            return checkpoint(candidate, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(SessionStore, "checkpoint", replace_slot)
            assert (
                store.submit(recover_record, store, "alice").result()["content"]
                == "old"
            )
        assert store.submit(store.state.get, "alice:open_text").result() == new
        assert store.submit(recover_record, store, "alice").result()["content"] == "new"
        assert store.submit(store.state.get, "alice:open_text").result() is None
        assert [
            row["content"]
            for row in store.submit(store.get_events, "alice").result()
            if row["type"] == "text_chunk"
        ] == ["old", "new"]
    finally:
        store.close(update_status=False)


def test_normal_finalization_receipt_covers_its_shorter_crash_slot(
    tmp_path, monkeypatch
):
    store = _create_store(tmp_path)
    try:
        generation = _seed_generation_declaration(store)
        partial = store.submit(
            build_record, store, "alice", "prefix", generation, 1, 1, []
        ).result()
        full = {**deepcopy(partial), "text": "prefix plus uncheckpointed suffix"}
        store.submit(store.state.__setitem__, "alice:open_text", partial).result()
        store.submit(store.checkpoint, strict=True).result()
        delete = type(store.state).__delitem__

        def fail_clear(table, key):
            if table is store.state and key == "alice:open_text":
                raise OSError("clear failed after confirmed full segment")
            return delete(table, key)

        with monkeypatch.context() as patch:
            patch.setattr(type(store.state), "__delitem__", fail_clear)
            with pytest.raises(OSError, match="clear failed"):
                store.submit(
                    recover_record, store, "alice", full, finalize=None
                ).result()
        assert store.submit(store.state.get, "alice:open_text").result() == partial
        path = store.path
        store.close(update_status=False)
        reopened = SessionStore(path)
        try:
            receipt = reopened.submit(recover_record, reopened, "alice").result()
            assert receipt["content"] == full["text"]
            assert (
                reopened.submit(reopened.state.get, "alice:open_text").result() is None
            )
            assert (
                len(
                    [
                        r
                        for r in reopened.submit(reopened.get_events, "alice").result()
                        if r["type"] == "text_chunk"
                    ]
                )
                == 1
            )
            assert (
                reopened.submit(
                    recover_record, reopened, "alice", full, finalize=None
                ).result()
                == receipt
            )
        finally:
            reopened.close(update_status=False)
    finally:
        if not store._closed:
            store.close(update_status=False)


def test_recovery_refuses_unproven_shorter_prefix_even_after_normal_completion(
    tmp_path,
):
    store = _create_store(tmp_path)
    try:
        generation = _seed_generation_declaration(store)
        record = store.submit(
            build_record, store, "alice", "whole text", generation, 1, 1, []
        ).result()
        store.submit(recover_record, store, "alice", record, finalize=None).result()
        forged = {**deepcopy(record), "text": "whole"}
        with pytest.raises(HistoryCompatibilityError, match="Conflicting"):
            store.submit(recover_record, store, "alice", forged).result()
        rows = store.submit(store.get_events, "alice").result()
        assert [r["content"] for r in rows if r["type"] == "text_chunk"] == [
            "whole text"
        ]
    finally:
        store.close(update_status=False)


def test_recover_record_empty_slot_returns_none(tmp_path):
    """recover_record returns None when open_text slot is empty."""
    store = _create_store(tmp_path)
    try:
        result = store.submit(recover_record, store, "alice").result()
        assert result is None
    finally:
        store.close()
