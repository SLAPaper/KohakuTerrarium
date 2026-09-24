"""Production response origins must be durable before the provider is called."""

import asyncio
import threading

import pytest

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


async def _creature(engine, tmp_path, fmt, provider):
    config = tmp_path / "agent.yaml"
    config.write_text(
        f"name: alice\nsystem_prompt: Reply\ntool_format: {fmt}\ninput: {{type: none}}\noutput: {{type: stdout}}\n"
    )
    creature = await engine.add_creature(
        config, llm=provider, io="headless", start=True
    )
    return creature, config


@pytest.mark.parametrize("fmt", ["native", "bracket"])
@pytest.mark.parametrize("failure", ["append", "flush"])
async def test_persistence_failure_blocks_real_provider(
    tmp_path, fmt, failure, monkeypatch
):
    provider = ScriptedLLM(["must not call"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature, _ = await _creature(engine, tmp_path, fmt, provider)
        store = creature.agent.session_store
        original = store.append_event
        entered = False

        def append(namespace, kind, data, **kwargs):
            nonlocal entered
            if kind == "assistant_generation_start":
                entered = True
                if failure == "append":
                    raise OSError("origin append failed")
            return original(namespace, kind, data, **kwargs)

        flush_original = store.flush

        def flush():
            if entered and failure == "flush":
                raise OSError("origin flush failed")
            return flush_original()

        with monkeypatch.context() as patch:
            patch.setattr(store, "append_event", append)
            patch.setattr(store, "flush", flush)
            result = await creature.run("first", timeout=5, raise_on_error=False)
            assert not result.ok
            assert provider.call_count == 0
            assert "origin" in str(result.error)
        assert creature.agent.controller._active_generation is None
        assert (
            await creature.run("retry explicitly", timeout=5)
        ).text == "must not call"
        assert provider.call_count == 1


@pytest.mark.parametrize("fmt", ["native", "bracket"])
async def test_two_turns_have_unique_confirmed_origins_and_cold_resume(tmp_path, fmt):
    provider = ScriptedLLM(["first reply", "second reply"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature, config = await _creature(engine, tmp_path, fmt, provider)
        store = creature.agent.session_store
        sinks = [
            o
            for o in creature.agent.output_router._secondary_outputs
            if getattr(o, "_store", None) is store
        ]
        assert len(sinks) == 1
        sinks[0]._capture_activity = False
        for text in ("first", "second"):
            assert (await creature.run(text, timeout=5)).ok
        events = await store.run(store.get_events, "alice")
        origins = [e for e in events if e["type"] == "assistant_generation_start"]
        assert len(origins) == 2
        assert len({e["generation_id"] for e in origins}) == 2
        for origin, reply in zip(origins, ("first reply", "second reply")):
            answer = next(
                e
                for e in events
                if e["type"] == "text_chunk" and e.get("content") == reply
            )
            assert answer["generation_id"] == origin["generation_id"]
            assert answer["event_id"] > origin["event_id"]
            assert answer["turn_index"] == origin["turn_index"]
            assert origin["source"]["tool_format"] == fmt
        assert not any(
            e["type"] == "activity:assistant_generation_start" for e in events
        )
        snapshot = await store.run(store.load_conversation, "alice")
        assert [
            m["metadata"]["generation"]["generation_id"]
            for m in snapshot
            if m["role"] == "assistant"
        ] == [e["generation_id"] for e in origins]
        latest = [
            m
            for m in creature.agent.controller.conversation.get_messages()
            if m.role == "assistant"
        ][-1]
        latest.metadata["generation"]["source"]["model"] = "changed message"
        assert (
            creature.agent.controller._active_generation["source"]["model"]
            != "changed message"
        )
        path = store.path
    reopened = SessionStore(path)
    resumed = await Agent.build(config, llm=ScriptedLLM(["cold reply"]), io="headless")
    try:
        await inject_saved_state_async(resumed, reopened, "alice")
        await reopened.run(resumed.attach_session_store, reopened)
        await resumed.start()
        assert resumed.llm.call_count == 0
        assert [
            m.content
            for m in resumed.controller.conversation.get_messages()
            if m.role == "user"
        ] == ["first", "second"]
        assert (await resumed.run("continue", timeout=5)).text == "cold reply"
    finally:
        await resumed.stop()
        await asyncio.to_thread(reopened.close)


async def test_interrupt_accepted_origin_never_starts_provider(tmp_path, monkeypatch):
    provider = ScriptedLLM(["not started"])
    entered, release = threading.Event(), threading.Event()
    task = None
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature, _ = await _creature(engine, tmp_path, "native", provider)
        store = creature.agent.session_store
        append_original = store.append_event

        def append(namespace, kind, data, **kwargs):
            if kind == "assistant_generation_start":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("origin barrier unreleased")
            return append_original(namespace, kind, data, **kwargs)

        monkeypatch.setattr(store, "append_event", append)
        try:
            task = asyncio.create_task(
                creature.run("cancel me", timeout=10, raise_on_error=False)
            )
            assert await asyncio.to_thread(entered.wait, 5)
            assert provider.call_count == 0
            creature.agent.interrupt()
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            events = await store.run(store.get_events, "alice")
            assert sum(e["type"] == "assistant_generation_start" for e in events) == 1
            assert provider.call_count == 0
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
