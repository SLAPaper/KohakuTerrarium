"""Retired-file routing is published only after destination durability."""

import threading
from types import SimpleNamespace

import pytest

from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.errors import SessionNotResumableError
from kohakuterrarium.session.readonly_view import SessionReadView
from kohakuterrarium.session.resume_target import resolve_resume_path
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.terrarium.session_successor import (
    publish_successors,
    stage_successor,
)


def test_pending_unowned_source_uses_live_members_and_survives_failed_checkpoint(
    tmp_path, monkeypatch
):
    source = SessionStore(tmp_path / "old.kohakutr")
    target = SessionStore(tmp_path / "current.kohakutr")
    source.init_meta("old", "agent", "/cfg", str(tmp_path), ["alice", "ghost"])
    target.init_meta("current", "agent", "/cfg", str(tmp_path), ["alice"])
    source.meta["live_graph_manifest"] = {"creatures": [{"name": "alice"}]}
    engine = SimpleNamespace(
        _session_stores={"current": target},
        _creatures={},
        _topology=SimpleNamespace(
            graphs={"current": SimpleNamespace(creature_ids=set())}
        ),
    )
    try:
        stage_successor(engine, source, [target])
        marker = source.meta["resume_successor"]
        assert marker["agents"] == ["alice"]
        assert marker["state"] == "pending"
        assert not source._closed
        with monkeypatch.context() as patch:
            patch.setattr(
                target,
                "checkpoint",
                lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
            )
            with pytest.raises(OSError, match="disk full"):
                publish_successors(engine, "current")
        with pytest.raises(SessionNotResumableError, match="pending"):
            resolve_resume_path(source.path)
        publish_successors(engine, "current")
        assert source.meta["resume_successor"]["state"] == "ready"
        assert resolve_resume_path(source.path) == tmp_path / "current.kohakutr"
        assert not source._closed
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_live_source_ready_checkpoint_runs_on_source_worker(tmp_path, monkeypatch):
    source = SessionStore(tmp_path / "source-affinity.kohakutr")
    target = SessionStore(tmp_path / "target-affinity.kohakutr")
    engine = SimpleNamespace(
        _session_stores={"current": target},
        _creatures={},
        _topology=SimpleNamespace(
            graphs={"current": SimpleNamespace(creature_ids=set())}
        ),
    )
    try:
        stage_successor(engine, source, [target])
        worker = source.submit(threading.get_ident).result(timeout=3)
        checkpoint = source.checkpoint
        observed = []

        def guarded_checkpoint(**kwargs):
            observed.append(threading.get_ident())
            assert observed[-1] == worker
            checkpoint(**kwargs)

        monkeypatch.setattr(source, "checkpoint", guarded_checkpoint)
        publish_successors(engine, "current")
        assert observed == [worker]
        assert source.meta["resume_successor"]["state"] == "ready"
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_ready_receipt_lost_before_registry_cleanup_retries_idempotently(tmp_path):
    source = SessionStore(tmp_path / "receipt-source.kohakutr")
    target = SessionStore(tmp_path / "receipt-target.kohakutr")
    engine = SimpleNamespace(
        _session_stores={"current": target},
        _creatures={},
        _topology=SimpleNamespace(
            graphs={"current": SimpleNamespace(creature_ids=set())}
        ),
    )
    try:
        stage_successor(engine, source, [target])
        pending = list(engine._pending_session_successors)
        publish_successors(engine, "current")
        engine._pending_session_successors[:] = pending
        publish_successors(engine, "current")
        assert not engine._pending_session_successors
        assert resolve_resume_path(source.path) == tmp_path / "receipt-target.kohakutr"
        assert source.meta["resume_successor"]["state"] == "ready"
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_destination_flush_failure_keeps_source_pending(tmp_path, monkeypatch):
    source = SessionStore(tmp_path / "flush-source.kohakutr")
    target = SessionStore(tmp_path / "flush-target.kohakutr")
    engine = SimpleNamespace(
        _session_stores={"current": target},
        _creatures={},
        _topology=SimpleNamespace(
            graphs={"current": SimpleNamespace(creature_ids=set())}
        ),
    )
    try:
        stage_successor(engine, source, [target])
        with monkeypatch.context() as patch:

            def failed_flush():
                raise OSError("destination cache flush failed")

            patch.setattr(target.conversation, "flush_cache", failed_flush)
            with pytest.raises(OSError, match="destination cache flush"):
                publish_successors(engine, "current")
            assert source.meta["resume_successor"]["state"] == "pending"
        publish_successors(engine, "current")
        assert source.meta["resume_successor"]["state"] == "ready"
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_stopped_member_checkpoint_preserves_explicit_branch_view(tmp_path):
    source = SessionStore(tmp_path / "view-source.kohakutr")
    target = SessionStore(tmp_path / "view-target.kohakutr")
    conversation = Conversation()
    conversation.append("user", "retained branch")
    agent = SimpleNamespace(
        config=SimpleNamespace(name="alice"),
        controller=SimpleNamespace(conversation=conversation),
        _turn_index=2,
        _branch_id=1,
        _parent_branch_path=[[1, 1]],
        _branch_view={1: 1, 4: 2},
    )
    engine = SimpleNamespace(
        _session_stores={"current": target},
        _creatures={"alice": SimpleNamespace(agent=agent)},
        _topology=SimpleNamespace(
            graphs={"current": SimpleNamespace(creature_ids={"alice"})}
        ),
    )
    try:
        stage_successor(engine, source, [target])
        publish_successors(engine, "current")
        assert target.state["alice:snapshot_branch"]["branch_view"] == {
            1: 1,
            2: 1,
            4: 2,
        }
        assert target.load_conversation("alice")[0]["content"] == "retained branch"
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_closed_source_publication_reopens_and_releases_writer(tmp_path):
    source = SessionStore(tmp_path / "closed-source.kohakutr", writer_lock=True)
    target = SessionStore(tmp_path / "current-source.kohakutr", writer_lock=True)
    engine = SimpleNamespace(
        _session_stores={"current": target},
        _creatures={},
        _topology=SimpleNamespace(
            graphs={"current": SimpleNamespace(creature_ids=set())}
        ),
    )
    try:
        stage_successor(engine, source, [target])
        source.close(update_status=False)
        publish_successors(engine, "current")
        assert source._closed
        with SessionReadView(source.path) as view:
            assert view.get("meta", "resume_successor")["state"] == "ready"
        assert resolve_resume_path(source.path) == tmp_path / "current-source.kohakutr"
        reopened = SessionStore(source.path, writer_lock=True)
        try:
            assert reopened.meta["resume_successor"]["state"] == "ready"
        finally:
            reopened.close(update_status=False)
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_changed_destination_identity_never_publishes_ready(tmp_path):
    source = SessionStore(tmp_path / "source.kohakutr")
    target = SessionStore(tmp_path / "target.kohakutr")
    engine = SimpleNamespace(
        _session_stores={"current": target},
        _creatures={},
        _topology=SimpleNamespace(
            graphs={"current": SimpleNamespace(creature_ids=set())}
        ),
    )
    try:
        stage_successor(engine, source, [target])
        target.meta["conversation_id"] = "replacement-identity"
        with pytest.raises(SessionNotResumableError, match="identity"):
            publish_successors(engine, "current")
        assert source.meta["resume_successor"]["state"] == "pending"
        with pytest.raises(SessionNotResumableError, match="pending"):
            resolve_resume_path(source.path)
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_legacy_destinations_receive_durable_distinct_successor_identities(tmp_path):
    source = SessionStore(tmp_path / "parent.kohakutr")
    targets = [SessionStore(tmp_path / f"child-{i}.kohakutr") for i in range(2)]
    engine = SimpleNamespace()
    try:
        stage_successor(engine, source, targets, kind="split")
        refs = source.meta["resume_successor"]["targets"]
        assert all(ref["conversation_id"] for ref in refs)
        assert len({ref["conversation_id"] for ref in refs}) == 2
        for target, ref in zip(targets, refs):
            assert target.meta["conversation_id"] == ref["conversation_id"]
            with SessionReadView(target.path) as view:
                assert view.get("meta", "conversation_id") == ref["conversation_id"]
    finally:
        source.close(update_status=False)
        for target in targets:
            target.close(update_status=False)
