"""Live model switching must confirm durable intent before publishing bindings."""

import asyncio
import threading

import pytest
from kohakuvault import KVault

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.core import agent_model
from kohakuterrarium.core.model_switch_runtime import (
    require_model_mutation_available,
)
from kohakuterrarium.errors import (
    ConflictError,
    HistoryCompatibilityError,
    SessionError,
)
from kohakuterrarium.llm.generation import generation_payload
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


class _OverflowFailure(ScriptedLLM):
    def __setattr__(self, name, value):
        if name == "_overflow_rescue":
            raise RuntimeError("overflow callback rejected")
        super().__setattr__(name, value)


@pytest.mark.parametrize("fail", [False, True])
async def test_live_switch_publishes_only_after_control_confirmation(
    tmp_path, monkeypatch, fail
):
    config = tmp_path / "agent.yaml"
    config.write_text(
        "name: alice\nsystem_prompt: Reply\ninput: {type: none}\noutput: {type: stdout}\n"
    )
    old = ScriptedLLM(["old response"])
    replacement = ScriptedLLM(["new response"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(config, llm=old, io="headless", start=True)
        store = creature.agent.session_store
        put = KVault.put
        inspected = []

        def write(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                inspected.append(creature.agent.llm is old)
                if fail:
                    raise OSError("control rejected")
            return put(table, key, value, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(KVault, "put", write)
            if fail:
                with pytest.raises(OSError, match="control rejected"):
                    await creature.agent.switch_model_async("openai/gpt-5.4-nano")
            else:
                assert (
                    await creature.agent.switch_model_async("openai/gpt-5.4-nano")
                    == "openai/gpt-5.4-nano"
                )
        assert inspected == [True]
        assert creature.agent.llm is (old if fail else replacement)
        assert creature.agent.controller.llm is creature.agent.llm
        controls = [
            event
            for event in await store.run(store.get_events, "alice")
            if event["type"] == "model_switch"
        ]
        assert len(controls) == (0 if fail else 1)
        expected = "old response" if fail else "new response"
        assert (await creature.run("continue after switch", timeout=5)).text == expected


def _config(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: alice\nsystem_prompt: Reply\ninput: {type: none}\noutput: {type: stdout}\n"
    )
    return path


async def _creature(tmp_path, llm):
    engine = Terrarium(session_dir=tmp_path / "sessions")
    await engine.__aenter__()
    creature = await engine.add_creature(
        _config(tmp_path), llm=llm, io="headless", start=True
    )
    return engine, creature


async def test_native_chat_control_records_route_and_old_binding(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    old.backend_key = "old-key"
    old.backend_type = "openai"
    old.provider_name = "openai"
    old.model = "old"
    replacement = ScriptedLLM(["new response"])
    replacement.backend_key = "old-key"
    replacement.backend_type = "openai"
    replacement.provider_name = "openai"
    replacement.model = "new"
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    try:
        creature.agent.controller.conversation.append(
            "assistant",
            None,
            metadata={"generation": generation_payload(old, "native")},
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        )
        creature.agent.controller.conversation.append("tool", "ok", tool_call_id="c1")
        seen = []
        put = KVault.put

        def write(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                seen.append(creature.agent.llm is old)
            return put(table, key, value, **kwargs)

        monkeypatch.setattr(KVault, "put", write)
        await creature.agent.switch_model_async("openai/gpt-5.4-nano")
        controls = [
            event
            for event in await creature.agent.session_store.run(
                creature.agent.session_store.get_events, "alice"
            )
            if event["type"] == "model_switch"
        ]
        assert seen == [True]
        assert controls[0]["classification"] == "native"
        assert controls[0]["schema"] == 1
        assert controls[0]["to_identifier"] == "openai/gpt-5.4-nano"
        assert controls[0]["from_source"]["backend_key"] == "old-key"
        assert controls[0]["to_source"]["backend_key"] == "old-key"
        assert (await creature.run("after native", timeout=5)).text == "new response"
    finally:
        await engine.__aexit__(None, None, None)


async def test_private_history_refuses_before_event(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    replacement = ScriptedLLM(["new response"])
    old.model = "old-private-context"
    replacement.model = "new-private-context"
    built = []

    def factory(name):
        built.append(replacement)
        return replacement

    monkeypatch.setattr(agent_model, "create_llm_from_profile_name", factory)
    engine, creature = await _creature(tmp_path, old)
    try:
        creature.agent.controller.conversation.append(
            "assistant", "secret", extra_fields={"reasoning_state": "opaque"}
        )
        with pytest.raises(HistoryCompatibilityError):
            await creature.agent.switch_model_async("openai/gpt-5.4-nano")
        events = await creature.agent.session_store.run(
            creature.agent.session_store.get_events, "alice"
        )
        assert not any(event.get("type") == "model_switch" for event in events)
        assert creature.agent.llm is old
        assert built == [replacement]
        assert (await creature.run("still old", timeout=5)).text == "old response"
    finally:
        await engine.__aexit__(None, None, None)


async def test_tool_prepare_failure_leaves_old_provider(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    replacement = ScriptedLLM(["new response"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )

    def broken(*args, **kwargs):
        raise RuntimeError("tool prepare failed")

    monkeypatch.setattr("kohakuterrarium.core.model_switch_runtime.prepare", broken)
    engine, creature = await _creature(tmp_path, old)
    try:
        with pytest.raises(RuntimeError, match="tool prepare failed"):
            await creature.agent.switch_model_async("openai/gpt-5.4-nano")
        events = await creature.agent.session_store.run(
            creature.agent.session_store.get_events, "alice"
        )
        assert not any(event.get("type") == "model_switch" for event in events)
        assert creature.agent.llm is old
        assert (await creature.run("unchanged", timeout=5)).text == "old response"
    finally:
        await engine.__aexit__(None, None, None)


async def test_delayed_write_does_not_call_replacement(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    replacement = ScriptedLLM(["new response"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    try:
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        put = KVault.put

        def write(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5), "control write was not released"
            return put(table, key, value, **kwargs)

        monkeypatch.setattr(KVault, "put", write)
        task = asyncio.create_task(
            creature.agent.switch_model_async("openai/gpt-5.4-nano")
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        assert creature.agent.llm is old
        assert replacement.call_count == 0
        queued = asyncio.create_task(creature.run("queued during switch", timeout=5))
        await asyncio.sleep(0.05)
        assert not queued.done()
        release.set()
        assert await asyncio.wait_for(task, timeout=5) == "openai/gpt-5.4-nano"
        assert (await queued).text == "new response"
        assert replacement.call_count == 1
    finally:
        release.set()
        await engine.__aexit__(None, None, None)


async def test_repeated_cancel_of_accepted_switch_finishes_and_reraises(
    tmp_path, monkeypatch
):
    old = ScriptedLLM(["old response"])
    replacement = ScriptedLLM(["new response"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    try:
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        put = KVault.put

        def write(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5), "control write was not released"
            return put(table, key, value, **kwargs)

        monkeypatch.setattr(KVault, "put", write)
        first = asyncio.create_task(
            creature.agent.switch_model_async("openai/gpt-5.4-nano")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        for _ in range(3):
            first.cancel()
            await asyncio.sleep(0)
            assert (
                not first.done()
            ), "accepted switch must join before cancellation returns"
        assert creature.agent.llm is old
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=5)
        assert creature.agent.llm is replacement
        controls = [
            event
            for event in await creature.agent.session_store.run(
                creature.agent.session_store.get_events, "alice"
            )
            if event["type"] == "model_switch"
        ]
        assert len(controls) == 1
    finally:
        release.set()
        await engine.__aexit__(None, None, None)


async def test_ambiguous_commit_fences_next_mutation(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    replacement = ScriptedLLM(["new response"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    put = KVault.put
    try:

        def write(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                divergent = {**value, "to_identifier": "openai/other"}
                put(table, key, divergent, **kwargs)
                raise OSError("write ack lost")
            return put(table, key, value, **kwargs)

        monkeypatch.setattr(KVault, "put", write)
        with pytest.raises(SessionError, match="Ambiguous"):
            await creature.agent.switch_model_async("openai/gpt-5.4-nano")
        assert creature.agent.llm is old
        with pytest.raises(ConflictError, match="uncertain"):
            require_model_mutation_available(creature.agent)
        result = await creature.run(
            "must not continue", timeout=5, raise_on_error=False
        )
        assert not result.ok
        assert old.call_count == 0
    finally:
        await engine.__aexit__(None, None, None)


async def test_stop_joins_accepted_switch_before_provider_close(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    replacement = ScriptedLLM(["new response"])
    closed = []
    original = ScriptedLLM.close

    async def track(self):
        closed.append(self)
        return await original(self)

    monkeypatch.setattr(ScriptedLLM, "close", track)
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    try:
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        put = KVault.put

        def write(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5), "control write was not released"
            return put(table, key, value, **kwargs)

        monkeypatch.setattr(KVault, "put", write)
        task = asyncio.create_task(
            creature.agent.switch_model_async("openai/gpt-5.4-nano")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        stopping = asyncio.create_task(creature.agent.stop())
        await asyncio.sleep(0.05)
        assert not stopping.done()
        assert closed == []
        with pytest.raises(ConflictError):
            await creature.agent.switch_model_async("openai/gpt-5.4-nano")
        release.set()
        assert await asyncio.wait_for(task, timeout=5) == "openai/gpt-5.4-nano"
        await asyncio.wait_for(stopping, timeout=5)
        assert replacement in closed
        assert old in closed
    finally:
        release.set()
        await engine.__aexit__(None, None, None)


async def test_cold_resume_uses_confirmed_provider(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    created = []

    def factory(name):
        provider = ScriptedLLM([f"from {name}"])
        provider.model = name
        created.append(provider)
        return provider

    monkeypatch.setattr(agent_model, "create_llm_from_profile_name", factory)
    engine, creature = await _creature(tmp_path, old)
    path = creature.agent.session_store.path
    try:
        assert (await creature.run("before", timeout=5)).text == "old response"
        assert await creature.agent.switch_model_async("openai/gpt-5.4-nano")
    finally:
        await engine.__aexit__(None, None, None)
    resumed = ScriptedLLM(["must not use bootstrap provider"])
    reopened = SessionStore(path)
    again = await Agent.build(_config(tmp_path), llm=resumed, io="headless")
    try:
        await inject_saved_state_async(again, reopened, "alice")
        await reopened.run(again.attach_session_store, reopened)
        assert again.llm is created[-1]
        assert again.llm is not resumed
        await again.start()
        assert (
            await again.run("continue", timeout=5)
        ).text == "from openai/gpt-5.4-nano"
        assert resumed.call_count == 0
        assert [
            m.content
            for m in again.controller.conversation.get_messages()
            if m.role == "user"
        ] == ["before", "continue"]
    finally:
        await again.stop()
        await asyncio.to_thread(reopened.close)


async def test_overflow_preparation_failure_never_commits(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    replacement = _OverflowFailure(["unused"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    try:
        with pytest.raises(RuntimeError, match="overflow callback rejected"):
            await creature.agent.switch_model_async("openai/gpt-5.4-nano")
        events = await creature.agent.session_store.run(
            creature.agent.session_store.get_events, "alice"
        )
        assert not any(event["type"] == "model_switch" for event in events)
        assert creature.agent.llm is old
    finally:
        await engine.__aexit__(None, None, None)


async def test_admission_reserves_before_accepted_task_starts(tmp_path, monkeypatch):
    old = ScriptedLLM(["old"])
    replacement = ScriptedLLM(["new"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    task = asyncio.create_task(creature.agent.switch_model_async("openai/gpt-5.4-nano"))
    try:
        await asyncio.sleep(0)
        assert creature.agent._model_switch_task is not None
        with pytest.raises(ConflictError, match="Model switch"):
            creature.agent.update_system_prompt("must not mutate", replace=True)
        assert await task == "openai/gpt-5.4-nano"
    finally:
        await asyncio.gather(task, return_exceptions=True)
        await engine.__aexit__(None, None, None)


async def test_compact_fallback_uses_prepared_provider(tmp_path, monkeypatch):
    old = ScriptedLLM(["old"])
    replacement = ScriptedLLM(["new"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    monkeypatch.setattr(Agent, "_build_compact_llm", lambda self, config: self.llm)
    try:
        await creature.agent.switch_model_async("openai/gpt-5.4-nano")
        assert creature.agent.compact_manager._llm is replacement
    finally:
        await engine.__aexit__(None, None, None)


async def test_capture_does_not_read_store_on_loop(tmp_path, monkeypatch):
    old = ScriptedLLM(["old"])
    replacement = ScriptedLLM(["new"])
    monkeypatch.setattr(
        agent_model, "create_llm_from_profile_name", lambda name: replacement
    )
    engine, creature = await _creature(tmp_path, old)
    store = creature.agent.session_store
    get_events = store.get_events
    reads = []

    def observe(*args, **kwargs):
        reads.append(threading.get_ident())
        return get_events(*args, **kwargs)

    monkeypatch.setattr(store, "get_events", observe)
    try:
        await creature.agent.switch_model_async("openai/gpt-5.4-nano")
        assert threading.get_ident() not in reads
    finally:
        await engine.__aexit__(None, None, None)


async def test_live_sync_switch_rejects_before_factory(tmp_path, monkeypatch):
    old = ScriptedLLM(["old response"])
    calls = []
    monkeypatch.setattr(
        agent_model,
        "create_llm_from_profile_name",
        lambda name: calls.append(name),
    )
    engine, creature = await _creature(tmp_path, old)
    try:
        with pytest.raises(ConflictError, match="switch_model_async"):
            creature.agent.switch_model("openai/gpt-5.4-nano")
        assert calls == []
        assert creature.agent.llm is old
    finally:
        await engine.__aexit__(None, None, None)
