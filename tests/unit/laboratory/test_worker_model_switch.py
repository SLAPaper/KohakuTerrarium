"""Worker model switching uses live admission and confirms durable intent."""

import asyncio

import pytest
from kohakuvault import KVault

from kohakuterrarium import Terrarium
from kohakuterrarium.core import agent_model
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.regeneration_reservation import reserve_regeneration
from kohakuterrarium.laboratory._internal.app import AppMessage
from kohakuterrarium.laboratory.adapters.terrarium_runtime import (
    TerrariumRuntimeAdapter,
)
from kohakuterrarium.testing.llm import ScriptedLLM


class Registrar:
    client_id = None

    def register_app_extension(self, namespace, handler):
        self.handler = handler


def request(creature):
    return AppMessage(
        namespace=TerrariumRuntimeAdapter.NAMESPACE,
        type="switch_model",
        body={"creature_id": creature.creature_id, "model": "openai/gpt-5.4-nano"},
        sender_node="controller",
        request_id="switch-rpc",
        in_reply_to=None,
    )


@pytest.mark.parametrize("fail", [False, True])
async def test_live_worker_switch_records_before_publication_and_preserves_failed_binding(
    tmp_path, monkeypatch, fail
):
    old, new = ScriptedLLM(["old reply"]), ScriptedLLM(["new reply"])
    monkeypatch.setattr(agent_model, "create_llm_from_profile_name", lambda _: new)
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=old, io="headless", start=True
        )
        adapter = TerrariumRuntimeAdapter(engine, Registrar())
        store, put, observed = creature.agent.session_store, KVault.put, []

        def write(table, key, value, **kwargs):
            if isinstance(value, dict) and value.get("type") == "model_switch":
                observed.append(creature.agent.llm is old)
                if fail:
                    raise OSError("worker model control write failed")
            return put(table, key, value, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(KVault, "put", write)
            if fail:
                assert await adapter._dispatch(request(creature)) == {
                    "error": {
                        "kind": "engine",
                        "message": "worker model control write failed",
                    }
                }
            else:
                assert await adapter._dispatch(request(creature)) == {
                    "model": "openai/gpt-5.4-nano"
                }
        assert observed == [True]
        assert creature.agent.llm is (old if fail else new)
        assert creature.agent.controller.llm is creature.agent.llm
        rows = await store.run(store.get_events, "alice")
        assert sum(row["type"] == "model_switch" for row in rows) == (0 if fail else 1)
        assert (await creature.run("continue", timeout=3)).text == (
            "old reply" if fail else "new reply"
        )
        assert (old.call_count, new.call_count) == ((1, 0) if fail else (0, 1))


async def test_foreign_worker_switch_cannot_bypass_compound_reserve_regeneration(
    tmp_path, monkeypatch
):
    old, new = ScriptedLLM(["old reply"]), ScriptedLLM(["new reply"])
    prepared = []

    def replacement(selector):
        prepared.append(selector)
        return new

    monkeypatch.setattr(agent_model, "create_llm_from_profile_name", replacement)
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=old, io="headless", start=True
        )
        adapter = TerrariumRuntimeAdapter(engine, Registrar())
        store = creature.agent.session_store
        before = await store.run(store.get_events, "alice")
        with reserve_regeneration(creature.agent):
            response = await asyncio.create_task(adapter._dispatch(request(creature)))
            assert response["error"]["kind"] == "conflict"
        assert prepared == []
        assert creature.agent.llm is old and creature.agent.controller.llm is old
        assert await store.run(store.get_events, "alice") == before
        assert (await creature.run("continue", timeout=3)).text == "old reply"
        assert old.call_count == 1 and new.call_count == 0
