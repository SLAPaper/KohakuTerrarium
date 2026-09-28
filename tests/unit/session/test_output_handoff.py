"""Async output consumers wait for topology handoff before choosing their sink."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.modules.output.event import OutputEvent
from kohakuterrarium.llm.generation import generation_payload
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.session.output import SessionOutput
from kohakuterrarium.session.store import SessionStore


async def test_unrelated_secondary_sink_is_not_redirected_to_primary_store(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            io="headless",
            start=True,
        )
        secondary = SessionStore(tmp_path / "secondary.kohakutr")
        try:
            sink = SessionOutput("alice", secondary, creature.agent)
            await sink.write("secondary only")
            await sink.flush()
            rows = await secondary.run(secondary.get_events, "alice")
            assert [row["content"] for row in rows if row["type"] == "text_chunk"] == [
                "secondary only"
            ]
            await creature.agent._session_output.flush()
            rows = await creature.agent.session_store.run(
                creature.agent.session_store.get_events, "alice"
            )
            assert not any(row.get("content") == "secondary only" for row in rows)
        finally:
            secondary.close(update_status=False)


@pytest.mark.parametrize("event_kind", ["activity", "terminal", "scratchpad"])
async def test_sync_activity_defers_before_store_submission(
    tmp_path, monkeypatch, event_kind
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        await alice.run("question")
        agent, source = alice.agent, alice.agent.session_store
        sink = agent._session_output
        context = capture_tool_context(agent, "read_sync", "direct", "call-sync")
        context["job_id"] = "read_sync"
        flush, append = type(source).flush, type(source).append_event
        once, destinations = False, []

        def held(store):
            nonlocal once
            if store is source and not once:
                once = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(6)
            return flush(store)

        def appended(store, namespace, kind, data, **kwargs):
            if event_kind != "scratchpad" and kind == (
                "tool_result" if event_kind == "terminal" else "context_cleared"
            ):
                destinations.append(store)
            return append(store, namespace, kind, data, **kwargs)

        monkeypatch.setattr(type(source), "flush", held)
        monkeypatch.setattr(type(source), "append_event", appended)
        save_state = type(source).save_state

        def saved(store, *args, **kwargs):
            if (
                event_kind == "scratchpad"
                and kwargs.get("scratchpad", {}).get("late") == "value"
            ):
                destinations.append(store)
            return save_state(store, *args, **kwargs)

        monkeypatch.setattr(type(source), "save_state", saved)
        moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if event_kind == "scratchpad":
                await agent._session_state_owner.submit_internal(
                    lambda: agent.session.scratchpad.set("late", "value")
                )
                receipt = None
            elif event_kind == "terminal":
                receipt = sink.record_terminal_activity(
                    "tool_done", "[read] original", context
                )
            else:
                receipt = sink.on_activity_with_metadata(
                    "context_cleared", "original", {"messages_cleared": 1}
                )
            release.set()
            await asyncio.wait_for(moving, 8)
            if receipt is not None:
                await asyncio.wait_for(asyncio.wrap_future(receipt), 3)
            await agent._session_output.flush()
            assert destinations == [agent.session_store]
        finally:
            release.set()
            await asyncio.gather(moving, return_exceptions=True)


@pytest.mark.parametrize(
    "method",
    ["write_stream", "write", "emit", "_begin_generation", "emit_start", "emit_end"],
)
async def test_late_text_waits_for_handoff_and_uses_rebound_sink(
    tmp_path, monkeypatch, method
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["bob"]), io="headless", start=True
        )
        await alice.run("question")
        agent, source = alice.agent, alice.agent.session_store
        old_sink = agent._session_output
        active_generation = generation_payload(agent.llm, "bracket")
        await old_sink._begin_generation(active_generation)
        flush = type(source).flush
        once = False

        def held(store):
            nonlocal once
            if store is source and not once:
                once = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(6)
            return flush(store)

        monkeypatch.setattr(type(source), "flush", held)
        handoff = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
        writing = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            payload = (
                OutputEvent(type="text", content="late small fragment")
                if method == "emit"
                else "late small fragment"
            )
            if method == "_begin_generation":
                payload = generation_payload(agent.llm, "bracket")
            if method in ("emit_start", "emit_end"):
                payload = OutputEvent(
                    type=(
                        "processing_start"
                        if method == "emit_start"
                        else "processing_end"
                    )
                )
            writing = asyncio.create_task(
                getattr(old_sink, "emit" if method.startswith("emit_") else method)(
                    payload
                )
            )
            await agent._session_state_owner.submit_internal(lambda: None)
            assert not writing.done()
            release.set()
            await asyncio.wait_for(handoff, 8)
            await asyncio.wait_for(writing, 3)
            await agent._session_output.flush()
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            if method == "_begin_generation":
                assert (
                    sum(
                        row.get("generation_id") == payload["generation_id"]
                        for row in rows
                    )
                    == 1
                )
                assert agent._session_output._generation_id == payload["generation_id"]
            elif method in ("emit_start", "emit_end"):
                assert sum(row["type"] == payload.type for row in rows) == 2
                assert not agent._session_output._inflight_output_calls
            else:
                fragments = [
                    row for row in rows if row.get("content") == "late small fragment"
                ]
                assert len(fragments) == 1
                assert (
                    fragments[0]["generation_id"] == active_generation["generation_id"]
                )
            assert agent.session_store is not source
        finally:
            release.set()
            await asyncio.gather(
                handoff, *([writing] if writing else []), return_exceptions=True
            )
