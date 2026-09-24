"""Test-only pre-provider durability gate exercised through a real Agent turn."""

import asyncio
import threading
from copy import deepcopy

import pytest

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.core.session_state_owner import SessionStateOwner
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


class _GenerationGate:
    def __init__(self, owner, store, agent):
        self.owner, self.store, self.agent = owner, store, agent
        self.pending = None
        self.confirmed = []

    def reserve(self):
        if self.pending is not None:
            raise ConflictError("generation persistence pending")
        self.pending = object()
        return self.pending, {
            "provider": "scripted",
            "turn": self.agent._turn_index,
            "branch": self.agent._branch_id,
        }

    async def before_provider(self):
        token, payload = await self.owner.submit(self.reserve)
        try:

            def persist():
                event = self.store.append_event(
                    self.agent.config.name, "probe_generation_start", deepcopy(payload)
                )
                self.store.flush()
                return event

            record = await self.store.run(persist)
            await self.owner.submit_internal(lambda: self.confirmed.append(record))
        finally:
            await self.owner.submit_internal(lambda: self.release(token))

    def release(self, token):
        assert self.pending is token
        self.pending = None


class _GatedLLM(ScriptedLLM):
    def __init__(self):
        super().__init__(["confirmed reply"])
        self.gate = None

    async def chat(self, messages, **kwargs):
        await self.gate.before_provider()
        async for chunk in super().chat(messages, **kwargs):
            yield chunk


@pytest.mark.parametrize("fail", [False, True])
async def test_real_turn_waits_for_durable_gate_before_scripted_provider(
    tmp_path, fail
):
    config = tmp_path / "agent.yaml"
    config.write_text(
        "name: probe\nsystem_prompt: Reply\ninput: {type: none}\noutput: {type: stdout}\n"
    )
    llm = _GatedLLM()
    owner = SessionStateOwner(name="generation-probe")
    owner.start()
    entered, release = threading.Event(), threading.Event()
    task = None
    try:
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            creature = await engine.add_creature(
                config, llm=llm, io="headless", start=True
            )
            store = creature.agent.session_store
            gate = _GenerationGate(owner, store, creature.agent)
            llm.gate = gate
            append = store.append_event

            def persist(*args, **kwargs):
                if args[1] == "probe_generation_start":
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("generation barrier not released")
                    if fail:
                        raise OSError("generation persistence failed")
                return append(*args, **kwargs)

            store.append_event = persist
            task = asyncio.create_task(
                creature.run("persist first", timeout=10, raise_on_error=False)
            )
            assert await asyncio.to_thread(entered.wait, 5)
            assert llm.call_count == 0
            assert gate.confirmed == []
            with pytest.raises(ConflictError, match="pending"):
                await owner.submit(gate.reserve)
            assert (
                await owner.submit(lambda: "unrelated callback") == "unrelated callback"
            )
            release.set()
            result = await task
            assert gate.pending is None
            events = await store.run(store.get_events, "probe")
            origins = [e for e in events if e["type"] == "probe_generation_start"]
            if fail:
                assert not result.ok
                assert llm.call_count == 0 and not origins and not gate.confirmed
            else:
                assert result.text == "confirmed reply"
                assert llm.call_count == 1 and len(origins) == len(gate.confirmed) == 1
                snapshot = await store.run(store.load_conversation, "probe")
                assert [m["content"] for m in snapshot if m["role"] != "system"] == [
                    "persist first",
                    "confirmed reply",
                ]
            path = store.path
        if not fail:
            reopened = SessionStore(path)
            cold_llm = ScriptedLLM(["cold continuation"])
            resumed = await Agent.build(config, llm=cold_llm, io="headless")
            try:
                await inject_saved_state_async(resumed, reopened, "probe")
                await reopened.run(resumed.attach_session_store, reopened)
                await resumed.start()
                assert cold_llm.call_count == 0
                assert [
                    m.content
                    for m in resumed.controller.conversation.get_messages()
                    if m.role != "system"
                ] == ["persist first", "confirmed reply"]
                assert (
                    await resumed.run("continue", timeout=5)
                ).text == "cold continuation"
            finally:
                await resumed.stop()
                await asyncio.to_thread(reopened.close)
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await owner.close()
