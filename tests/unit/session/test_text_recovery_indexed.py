"""Performance and indexed declaration verification tests for text recovery."""

from copy import deepcopy
from uuid import uuid4

import pytest

import kohakuterrarium.session.event_origin as event_origin_mod
import kohakuterrarium.session.text_recovery as text_recovery_mod
from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import HistoryCompatibilityError, SessionError
from kohakuterrarium.llm.generation import generation_payload
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.session.output import SessionOutput
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.text_buffer import OpenTextSegment
from kohakuterrarium.session.text_recovery import (
    build_record,
    recover_record,
    prove_generation_origin,
    validate_record,
)


class _FakeAgent:
    def __init__(self, *, turn=1, branch=1, path=None):
        self._turn_index = turn
        self._branch_id = branch
        self._parent_branch_path = path or []
        self.controller = None
        self.session = None


def _create_store(tmp_path, name="session.kohakutr"):
    path = tmp_path / name
    store = SessionStore(str(path))
    store.submit(event_origin_mod.store_uuid, store).result()
    return store


@pytest.mark.asyncio
async def test_subsequent_gated_writes_avoid_full_scans_after_confirmation(
    tmp_path, monkeypatch
):
    """After generation confirmation and index build, 2nd and 3rd gated writes do not scan full store."""
    store = _create_store(tmp_path)
    try:
        agent = _FakeAgent(turn=1, branch=1)
        out = SessionOutput("alice", store, agent, event_key_prefix="alice")
        payload = {
            "schema": 1,
            "generation_id": str(uuid4()),
            "source": {"provider_name": "scripted", "model": "test-model"},
        }
        await out._begin_generation(payload)

        # Warm the agent index so subsequent lookups are O(1)
        store.submit(store.get_event_by_id, "alice", 1).result()

        capture_scan_count = 0
        original_capture = event_origin_mod.capture_events

        def counting_capture(s):
            nonlocal capture_scan_count
            capture_scan_count += 1
            return original_capture(s)

        get_events_count = 0
        original_get_events = store.get_events

        def counting_get_events(agent_name, *args, **kwargs):
            nonlocal get_events_count
            get_events_count += 1
            return original_get_events(agent_name, *args, **kwargs)

        monkeypatch.setattr(text_recovery_mod, "capture_events", counting_capture)
        monkeypatch.setattr(event_origin_mod, "capture_events", counting_capture)
        monkeypatch.setattr(store, "get_events", counting_get_events)

        # 1st gated write
        out._ingest_text("A" * 550)
        await out.drain()

        # Reset counts before 2nd and 3rd gated writes
        capture_scan_count = 0
        get_events_count = 0

        # 2nd gated write
        out._ingest_text("B" * 550)
        await out.drain()

        # 3rd gated write
        out._ingest_text("C" * 550)
        await out.drain()

        assert (
            capture_scan_count == 0
        ), f"Expected 0 capture_events scans during 2nd/3rd gated writes, got {capture_scan_count}"
        assert (
            get_events_count == 0
        ), f"Expected 0 get_events scans during 2nd/3rd gated writes, got {get_events_count}"

        slot = store.submit(store.state.get, "alice:open_text").result()
        assert isinstance(slot, dict)
        assert slot["text"] == ("A" * 550) + ("B" * 550) + ("C" * 550)
        assert slot["generation"]["generation_id"] == payload["generation_id"]
    finally:
        store.close(update_status=False)


@pytest.mark.asyncio
async def test_corrupted_indexed_declaration_source_fails_gated_write(tmp_path):
    """Corrupting declaration source in the store fails subsequent gated writes."""
    store = _create_store(tmp_path)
    try:
        agent = _FakeAgent(turn=1, branch=1)
        out = SessionOutput("alice", store, agent, event_key_prefix="alice")
        payload = {
            "schema": 1,
            "generation_id": str(uuid4()),
            "source": {"provider_name": "scripted", "model": "test-model"},
        }
        await out._begin_generation(payload)

        # 1st gated write succeeds
        out._ingest_text("A" * 550)
        await out.drain()

        # Corrupt declaration source in the store
        def corrupt_source():
            for key in list(store.events.keys()):
                evt = dict(store.events[key])
                if evt.get("type") == "assistant_generation_start":
                    evt["source"] = {"provider_name": "scripted", "model": "corrupted"}
                    store.events[key] = evt
            store.flush()

        store.submit(corrupt_source).result()

        # 2nd gated write must fail with HistoryCompatibilityError
        with pytest.raises((HistoryCompatibilityError, SessionError)):
            out._ingest_text("B" * 550)
            await out.drain()
    finally:
        store.close(update_status=False)


@pytest.mark.asyncio
async def test_corrupted_indexed_declaration_genid_fails_gated_write(tmp_path):
    """Corrupting declaration generation_id in the store fails subsequent gated writes."""
    store = _create_store(tmp_path)
    try:
        agent = _FakeAgent(turn=1, branch=1)
        out = SessionOutput("alice", store, agent, event_key_prefix="alice")
        payload = {
            "schema": 1,
            "generation_id": str(uuid4()),
            "source": {"provider_name": "scripted", "model": "test-model"},
        }
        await out._begin_generation(payload)

        # 1st gated write succeeds
        out._ingest_text("A" * 550)
        await out.drain()

        # Corrupt generation_id in the store
        def corrupt_genid():
            for key in list(store.events.keys()):
                evt = dict(store.events[key])
                if evt.get("type") == "assistant_generation_start":
                    evt["generation_id"] = "corrupted-gen-id"
                    store.events[key] = evt
            store.flush()

        store.submit(corrupt_genid).result()

        # 2nd gated write must fail
        with pytest.raises((HistoryCompatibilityError, SessionError)):
            out._ingest_text("B" * 550)
            await out.drain()
    finally:
        store.close(update_status=False)


@pytest.mark.asyncio
async def test_foreign_store_proof_rejected(tmp_path):
    """A record or proof bound to a foreign store is rejected."""
    store_a = _create_store(tmp_path, "store_a.kohakutr")
    store_b = _create_store(tmp_path, "store_b.kohakutr")
    try:
        agent_a = _FakeAgent(turn=1, branch=1)
        out_a = SessionOutput("alice", store_a, agent_a, event_key_prefix="alice")
        payload = {
            "schema": 1,
            "generation_id": str(uuid4()),
            "source": {"provider_name": "scripted", "model": "test-model"},
        }
        await out_a._begin_generation(payload)
        out_a._ingest_text("Hello foreign store " * 30)
        await out_a.drain()

        slot_a = store_a.submit(store_a.state.get, "alice:open_text").result()
        assert slot_a is not None

        # Validating slot_a in store_b must be rejected
        with pytest.raises(HistoryCompatibilityError, match="[Ss]tore UUID"):
            store_b.submit(validate_record, store_b, "alice", slot_a).result()
    finally:
        store_a.close(update_status=False)
        store_b.close(update_status=False)


@pytest.mark.asyncio
async def test_one_gen_two_segments_chunk_seq_normal(tmp_path):
    """Two segments within one generation have monotonic chunk_seq and persist normally."""
    store = _create_store(tmp_path)
    try:
        agent = _FakeAgent(turn=1, branch=1)
        out = SessionOutput("alice", store, agent, event_key_prefix="alice")
        payload = {
            "schema": 1,
            "generation_id": str(uuid4()),
            "source": {"provider_name": "scripted", "model": "test-model"},
        }
        await out._begin_generation(payload)

        # Segment 1
        out._ingest_text("First segment content")
        out._flush_text_segment()
        await out.drain()

        # Segment 2
        out._ingest_text("Second segment content")
        out._flush_text_segment()
        await out.drain()

        events = store.submit(store.get_events, "alice").result()
        chunks = [e for e in events if e.get("type") == "text_chunk"]
        assert len(chunks) == 2
        assert chunks[0]["chunk_seq"] == 0
        assert chunks[0]["content"] == "First segment content"
        assert chunks[0]["generation_id"] == payload["generation_id"]

        assert chunks[1]["chunk_seq"] == 1
        assert chunks[1]["content"] == "Second segment content"
        assert chunks[1]["generation_id"] == payload["generation_id"]
    finally:
        store.close(update_status=False)


async def test_live_handoff_rebinds_generation_proof_before_repeated_text_gates(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM([]), io="headless", start=True
        )
        await alice.run("question", timeout=3)
        old_sink = alice.agent._session_output
        await old_sink._begin_generation(generation_payload(alice.agent.llm, "bracket"))
        origin = deepcopy(old_sink._generation_proof["declaration_origin"])
        source = alice.agent.session_store
        await engine.connect(alice, bob, channel="proof-transfer")
        sink = alice.agent._session_output
        destination = alice.agent.session_store
        assert destination is not source
        scans = []
        capture = text_recovery_mod.capture_events

        def observe(store):
            scans.append(store)
            return capture(store)

        with monkeypatch.context() as patch:
            patch.setattr(text_recovery_mod, "capture_events", observe)
            for value in ("A", "B", "C"):
                await old_sink.write_stream(value * 550)
                await sink.drain()
            assert scans == []
        record = await destination.run(destination.state.get, "alice:open_text")
        assert record["declaration_origin"] == origin
        assert record["store_uuid"] != origin[0]
        assert record["text"] == "A" * 550 + "B" * 550 + "C" * 550
        await sink.flush()
        rows = await destination.run(destination.get_events, "alice")
        assert sum(row.get("content") == record["text"] for row in rows) == 1


async def test_transferred_declaration_keeps_original_origin_and_current_store_binding(
    tmp_path,
):
    source = _create_store(tmp_path, "source.kohakutr")
    destination = _create_store(tmp_path, "destination.kohakutr")
    try:
        sink = SessionOutput("alice", source, _FakeAgent())
        await sink._begin_generation(
            {
                "schema": 1,
                "generation_id": str(uuid4()),
                "source": {"model": "confirmed"},
            }
        )
        generation = deepcopy(sink._generation_payload)
        original = deepcopy(sink._generation_proof["declaration_origin"])
        await destination.run(
            destination.append_event, "other", "message", {"content": "occupied id"}
        )
        await destination.run(
            event_origin_mod.append_imported,
            destination,
            "alice",
            "assistant_generation_start",
            {
                **deepcopy(generation),
                event_origin_mod.EVENT_ORIGIN: original,
                "turn_index": 1,
                "branch_id": 1,
                "parent_branch_path": [],
            },
        )
        await destination.run(destination.checkpoint, strict=True)
        proof = await destination.run(
            prove_generation_origin, destination, "alice", generation, 1, 1, []
        )
        assert proof["declaration_origin"] == original
        assert proof["event_id"] != original[2]
        assert proof["store_uuid"] != original[0]
        record = await destination.run(
            build_record,
            destination,
            "alice",
            "transferred partial",
            generation,
            1,
            1,
            [],
            ephemeral_proof=proof,
        )
        await destination.run(destination.state.__setitem__, "alice:open_text", record)
        receipt = await destination.run(recover_record, destination, "alice")
        assert receipt["content"] == "transferred partial"
        assert receipt["segment_origin"]["declaration_origin"] == original
        assert receipt["segment_origin"]["store_uuid"] == proof["store_uuid"]
        with pytest.raises(HistoryCompatibilityError, match="store UUID"):
            await source.run(validate_record, source, "alice", record)
    finally:
        source.close(update_status=False)
        destination.close(update_status=False)


def test_legacy_standalone_open_text_segment_unbound_compatibility(tmp_path):
    """Legacy standalone OpenTextSegment allows unbound writes when require_binding=False."""
    store = _create_store(tmp_path)
    try:
        # require_binding=False: unbound writes succeed and write plain strings
        seg_loose = OpenTextSegment(store, "bob", require_binding=False)
        seg_loose.append("Direct logging text " * 30)
        assert store.submit(store.state.get, "bob:open_text").result() == (
            "Direct logging text " * 30
        )
        assert seg_loose.take() == "Direct logging text " * 30

        # require_binding=True: unbound writes fail when checkpoint tripped
        seg_strict = OpenTextSegment(store, "carol", require_binding=True)
        with pytest.raises(SessionError, match="confirmed generation"):
            seg_strict.append("Direct logging text " * 30)
    finally:
        store.close(update_status=False)
