"""Snapshot bases retain their physical source prefix despite nonmonotonic IDs."""

import json
from copy import deepcopy
from uuid import uuid4

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.errors import ConflictError, HistoryCompatibilityError
from kohakuterrarium.core import compact_branch
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.session import output as session_output
from kohakuterrarium.session.event_origin import import_events
from kohakuterrarium.session.snapshot_coverage import (
    capture_snapshot_transfer,
    install_snapshot_transfer,
)
from kohakuterrarium.session.resume_branch import load_conversation_with_replay_fallback
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.snapshot_coverage import save_bound_snapshot
from kohakuterrarium.terrarium.session_coord import (
    merge_session_stores,
    _copy_session_into,
)


async def test_compact_event_after_snapshot_replaces_base_instead_of_appending_summary(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a1", "a2", "a3"]),
            io="headless",
            start=True,
        )
        for question in ("q1", "q2", "q3"):
            await creature.run(question)
        agent, store = creature.agent, creature.agent.session_store
        submit = compact_branch.submit_internal_session_state

        async def held(host, callback):
            result = await submit(host, callback)
            await store.run(
                store.append_event,
                "alice",
                "user_message",
                {"content": "late tail"},
                turn_index=4,
                branch_id=1,
            )
            return result

        monkeypatch.setattr(compact_branch, "submit_internal_session_state", held)
        agent.compact_manager.config.keep_recent_turns = 1
        agent.compact_manager._llm = ScriptedLLM(["summary"])
        await agent.compact_manager._run_compact()
        restored = await store.run(
            load_conversation_with_replay_fallback, store, "alice"
        )
        assert [m["content"] for m in restored if m["role"] != "system"] == [
            "summary",
            "q3",
            "a3",
            "late tail",
        ]


async def test_compact_snapshot_rejects_advance_after_owner_capture(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            io="headless",
            start=True,
        )
        await creature.run("question")
        agent = creature.agent
        store = agent.session_store
        before = await store.run(store.load_conversation, "alice")
        submit = compact_branch.submit_internal_session_state

        async def held(host, callback):
            result = await submit(host, callback)
            await store.run(
                store.append_event,
                "alice",
                "user_message",
                {"content": "late tail"},
                turn_index=2,
                branch_id=1,
            )
            return result

        monkeypatch.setattr(compact_branch, "submit_internal_session_state", held)
        with pytest.raises(ConflictError, match="prefix advanced"):
            await compact_branch.persist_compacted_async(
                store, "alice", agent, agent.controller.conversation, 1, 10
            )
        assert await store.run(store.load_conversation, "alice") == before
        restored = await store.run(
            load_conversation_with_replay_fallback, store, "alice"
        )
        assert restored[-1]["content"] == "late tail"


async def test_turn_snapshot_does_not_cover_event_written_after_owner_capture(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "held-snapshot.kohakutr")
    base = [{"role": "assistant", "content": "prior compacted base"}]
    try:
        await store.run(
            store.append_event,
            "alice",
            "user_message",
            {"content": "prefix"},
            turn_index=1,
            branch_id=1,
        )
        await store.run(save_bound_snapshot, store, "alice", base)
        sink = session_output.SessionOutput("alice", store, None)

        async def capture(_):
            await store.run(
                store.append_event,
                "alice",
                "user_message",
                {"content": "late tail"},
                turn_index=2,
                branch_id=1,
            )
            return deepcopy(base), None, {}

        monkeypatch.setattr(session_output, "capture_turn_snapshot", capture)
        await sink.on_processing_end()
        restored = await store.run(
            load_conversation_with_replay_fallback, store, "alice"
        )
        assert [message["content"] for message in restored] == [
            "prior compacted base",
            "late tail",
        ]
    finally:
        store.close(update_status=False)


def test_snapshot_base_and_selector_survive_displaced_copy_and_reopen(tmp_path):
    source = SessionStore(tmp_path / "source.kohakutr")
    padding = SessionStore(tmp_path / "padding.kohakutr")
    target = None
    try:

        def prepare():
            source.append_event(
                "alice",
                "user_message",
                {"event_id": 40, "content": "keep"},
                turn_index=1,
                branch_id=1,
            )
            save_bound_snapshot(
                source,
                "alice",
                [
                    {
                        "role": "user",
                        "content": "snapshot-only edited user",
                        "metadata": {
                            "event_id": 40,
                            "turn_index": 1,
                            "branch_id": 1,
                            "user_pair_id": "opaque-pair-token",
                        },
                    }
                ],
            )
            source.state["alice:snapshot_event_id"] = 40
            source.append_event(
                "alice",
                "user_message",
                {"event_id": 7, "content": "tail"},
                turn_index=2,
                branch_id=1,
            )
            source.flush()

        source.submit(prepare).result()
        padding.submit(padding.append_event, "bob", "probe", {"event_id": 900}).result()
        target = merge_session_stores([padding, source], tmp_path / "target.kohakutr")
        messages = target.submit(
            load_conversation_with_replay_fallback, target, "alice"
        ).result()
        assert [message["content"] for message in messages] == [
            "snapshot-only edited user",
            "tail",
        ]
        rows = target.submit(target.get_events, "alice").result()
        assert messages[0]["metadata"]["event_id"] == rows[0]["event_id"] != 40
        assert messages[0]["metadata"]["user_pair_id"] == "opaque-pair-token"
        target.close(update_status=False)
        target = SessionStore(tmp_path / "target.kohakutr")
        assert (
            target.submit(
                load_conversation_with_replay_fallback, target, "alice"
            ).result()
            == messages
        )
    finally:
        for store in (source, padding, target):
            if store is not None:
                store.close(update_status=False)


@pytest.mark.parametrize("foreign", [False, True])
def test_context_without_raw_source_keeps_content_without_false_local_selector(
    tmp_path, foreign
):
    source, target = SessionStore(tmp_path / "source.kohakutr"), SessionStore(
        tmp_path / "target.kohakutr"
    )
    try:
        source.submit(
            source.append_event,
            "alice",
            "user_message",
            {"content": "different local message"},
        ).result()
        source.submit(save_bound_snapshot, source, "alice", []).result()
        uid = source.submit(lambda: source.conversation["alice"]["store_uuid"]).result()
        origin = [str(uuid4()) if foreign else uid, "alice", 999]
        messages = [
            {
                "role": "user",
                "content": "external context only",
                "metadata": {"event_id": 1, "event_origin": origin},
            }
        ]
        source.submit(save_bound_snapshot, source, "alice", messages).result()
        if foreign:
            _copy_session_into(source, target)
            snapshot = target.submit(target.load_conversation, "alice").result()
            assert snapshot[0]["content"] == "external context only"
            assert snapshot[0]["metadata"]["event_origin"] == origin
            assert "event_id" not in snapshot[0]["metadata"]
        else:
            with pytest.raises(HistoryCompatibilityError, match="selector absent"):
                _copy_session_into(source, target)
            assert target.submit(target.load_conversation, "alice").result() is None
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


@pytest.mark.parametrize("damage", ["no_watermark", "nonmonotonic", "foreign"])
def test_unproven_legacy_snapshot_is_rejected_before_destination_copy(tmp_path, damage):
    source, target = SessionStore(tmp_path / "legacy.kohakutr"), SessionStore(
        tmp_path / "target.kohakutr"
    )
    try:
        source.submit(
            source.append_event,
            "alice",
            "user_message",
            {"content": "old", "event_id": 20},
        ).result()
        source.submit(
            source.save_conversation,
            "alice",
            [{"role": "assistant", "content": "legacy"}],
        ).result()
        if damage != "no_watermark":
            source.submit(
                lambda: source.state.__setitem__("alice:snapshot_event_id", 20)
            ).result()
        if damage == "nonmonotonic":
            source.submit(
                source.append_event,
                "alice",
                "user_message",
                {"content": "late", "event_id": 7},
            ).result()
        if damage == "foreign":
            batch, _ = source.submit(capture_snapshot_transfer, source).result()
            target.submit(import_events, target, batch).result()
            target.submit(
                target.save_conversation,
                "alice",
                [{"role": "assistant", "content": "foreign"}],
            ).result()
            target.submit(
                lambda: target.state.__setitem__("alice:snapshot_event_id", 20)
            ).result()
            with pytest.raises(HistoryCompatibilityError, match="proven local"):
                target.submit(capture_snapshot_transfer, target).result()
        else:
            with pytest.raises(HistoryCompatibilityError, match="proven local"):
                _copy_session_into(source, target)
            assert target.submit(target.get_events, "alice").result() == []
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


@pytest.mark.parametrize("encoded", [False, True])
def test_proven_local_legacy_watermark_migrates_without_resurrecting_prefix(
    tmp_path, encoded
):
    source, target = SessionStore(tmp_path / "legacy.kohakutr"), SessionStore(
        tmp_path / "target.kohakutr"
    )
    try:
        source.submit(
            source.append_event,
            "alice",
            "user_message",
            {"content": "old"},
            turn_index=1,
            branch_id=1,
        ).result()
        base = [{"role": "assistant", "content": "legacy summary"}]
        source.submit(
            source.save_conversation, "alice", json.dumps(base) if encoded else base
        ).result()
        source.submit(
            lambda: source.state.__setitem__("alice:snapshot_event_id", 1)
        ).result()
        source.submit(
            source.append_event,
            "alice",
            "user_message",
            {"content": "tail"},
            turn_index=2,
            branch_id=1,
        ).result()
        _copy_session_into(source, target)
        restored = target.submit(
            load_conversation_with_replay_fallback, target, "alice"
        ).result()
        assert [message["content"] for message in restored] == [
            "legacy summary",
            "tail",
        ]
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


@pytest.mark.parametrize("cleared", [False, True])
def test_transfer_replaces_stale_snapshot_and_keeps_uncovered_holes_across_hops(
    tmp_path, cleared
):
    source, target, third = [
        SessionStore(tmp_path / f"{name}.kohakutr")
        for name in ("source", "target", "third")
    ]
    try:
        source.submit(
            source.append_event,
            "alice",
            "user_message",
            {"content": "prefix"},
            turn_index=1,
            branch_id=1,
        ).result()
        base = [] if cleared else [{"role": "assistant", "content": "summary"}]
        source.submit(save_bound_snapshot, source, "alice", base).result()
        source.submit(
            source.append_event,
            "alice",
            "user_message",
            {"content": "tail"},
            turn_index=3,
            branch_id=1,
        ).result()
        target.submit(
            target.append_event,
            "alice",
            "user_message",
            {"content": "unrelated hole"},
            turn_index=2,
            branch_id=1,
        ).result()
        target.submit(
            save_bound_snapshot,
            target,
            "alice",
            [{"role": "assistant", "content": "stale"}],
        ).result()
        _copy_session_into(source, target)
        _copy_session_into(source, target)
        _copy_session_into(target, third)
        _copy_session_into(third, target)
        restored = target.submit(
            load_conversation_with_replay_fallback, target, "alice"
        ).result()
        assert [message["content"] for message in restored] == (
            [] if cleared else ["summary"]
        ) + ["unrelated hole", "tail"]
        source.submit(lambda: source.conversation.pop("alice", None)).result()
        _copy_session_into(source, target)
        assert target.submit(target.load_conversation, "alice").result() is None
    finally:
        for store in (source, target, third):
            store.close(update_status=False)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "destination", "metadata_type", "bad_origin"]
)
def test_transfer_rejects_invalid_receipts_before_snapshot_install(tmp_path, mutation):
    source, target = SessionStore(tmp_path / "a.kohakutr"), SessionStore(
        tmp_path / "b.kohakutr"
    )
    try:
        source.submit(
            source.append_event, "alice", "user_message", {"content": "source"}
        ).result()
        source.submit(
            save_bound_snapshot,
            source,
            "alice",
            [{"role": "user", "content": "source", "metadata": {"event_id": 1}}],
        ).result()
        batch, snapshots = source.submit(capture_snapshot_transfer, source).result()
        receipts = target.submit(import_events, target, batch).result()
        receipts = deepcopy(receipts)
        if mutation == "missing":
            receipts.clear()
        elif mutation == "duplicate":
            receipts.append(deepcopy(receipts[0]))
        elif mutation == "metadata_type":
            snapshots["alice"][0]["messages"][0]["metadata"] = "invalid"
        elif mutation == "bad_origin":
            snapshots["alice"][0]["messages"][0]["metadata"]["event_origin"] = [
                "invalid"
            ]
        else:
            uid, namespace, key, _ = receipts[0]["destination"]
            receipts[0]["destination"] = (uid, namespace, key, 999)
        target.submit(
            target.save_conversation,
            "alice",
            [{"role": "user", "content": "unchanged"}],
        ).result()
        with pytest.raises(HistoryCompatibilityError):
            target.submit(
                install_snapshot_transfer, target, batch, snapshots, receipts
            ).result()
        assert (
            target.submit(target.load_conversation, "alice").result()[0]["content"]
            == "unchanged"
        )
    finally:
        source.close(update_status=False)
        target.close(update_status=False)


def test_snapshot_tail_uses_physical_prefix_not_maximum_event_id(tmp_path):
    store = SessionStore(tmp_path / "snapshot.kohakutr")
    try:

        def write():
            for event_id, turn in ((40, 1), (7, 2)):
                store.append_event(
                    "alice",
                    "user_message",
                    {"event_id": event_id, "content": f"prefix-{turn}"},
                    turn_index=turn,
                    branch_id=1,
                )
            save_bound_snapshot(
                store,
                "alice",
                [{"role": "assistant", "content": "snapshot-only summary"}],
            )
            store.state["alice:snapshot_event_id"] = 40
            store.append_event(
                "alice",
                "user_message",
                {"event_id": 3, "content": "late lower-id tail"},
                turn_index=3,
                branch_id=1,
            )
            store.flush()

        store.submit(write).result()
        messages = store.submit(
            load_conversation_with_replay_fallback, store, "alice"
        ).result()
        assert [(m["role"], m["content"]) for m in messages] == [
            ("assistant", "snapshot-only summary"),
            ("user", "late lower-id tail"),
        ]
    finally:
        store.close(update_status=False)
    reopened = SessionStore(tmp_path / "snapshot.kohakutr")
    try:
        messages = reopened.submit(
            load_conversation_with_replay_fallback, reopened, "alice"
        ).result()
        assert [m["content"] for m in messages] == [
            "snapshot-only summary",
            "late lower-id tail",
        ]
    finally:
        reopened.close(update_status=False)
