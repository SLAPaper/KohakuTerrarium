"""Cold resume rejects unsupported snapshot provenance before installing state."""

import asyncio
from copy import deepcopy

import pytest

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize(
    "damage",
    [
        "orphan",
        "duplicate_result",
        "interleaved",
        "duplicate_announcement",
        "missing_type",
        "foreign_result",
        "malformed_pending",
    ],
)
async def test_native_snapshot_damage_rejected_before_sanitizing(tmp_path, damage):
    store = SessionStore(str(tmp_path / "native.kohakutr"))
    agent = await Agent.build(
        AgentConfig(name="fresh"), llm=ScriptedLLM(["must not run"]), io="headless"
    )
    call = {
        "id": "native",
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }
    announcement = {"role": "assistant", "content": "", "tool_calls": [call]}
    result = {
        "role": "tool",
        "content": "result",
        "tool_call_id": "native",
        "name": "read",
    }
    match damage:
        case "orphan":
            body = [result]
        case "duplicate_result":
            body = [announcement, result, deepcopy(result)]
        case "interleaved":
            body = [announcement, {"role": "user", "content": "unexpected"}, result]
        case "duplicate_announcement":
            body = [announcement, result, deepcopy(announcement), deepcopy(result)]
        case "missing_type":
            del call["type"]
            body = [announcement, result]
        case "foreign_result":
            result["tool_call_id"] = "not-announced"
            body = [announcement, result]
        case "malformed_pending":
            call["id"] = ""
            body = [announcement]
    try:
        await store.run(
            store.save_conversation,
            "alice",
            [
                {"role": "system", "content": "summary"},
                {"role": "user", "content": "question"},
                *body,
            ],
        )
        conversation = agent.controller.conversation
        with pytest.raises(HistoryCompatibilityError):
            await inject_saved_state_async(agent, store, "alice")
        assert agent.controller.conversation is conversation
        assert agent.config.name == "fresh"
        assert agent.llm.call_count == 0
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("partial", [False, True])
async def test_valid_pending_native_snapshot_is_not_discarded(tmp_path, partial):
    store = SessionStore(str(tmp_path / "pending.kohakutr"))
    agent = await Agent.build(
        AgentConfig(name="fresh"), llm=ScriptedLLM(["must not run"]), io="headless"
    )
    calls = [
        {
            "id": "native",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
    ]
    results = []
    if partial:
        calls.append({**deepcopy(calls[0]), "id": "pending"})
        results.append({"role": "tool", "content": "done", "tool_call_id": "native"})
    try:
        await store.run(
            store.save_conversation,
            "alice",
            [
                {"role": "system", "content": "summary"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "", "tool_calls": calls},
                *results,
            ],
        )
        await inject_saved_state_async(agent, store, "alice")
        messages = agent.controller.conversation.get_messages()
        assert messages[0].content == "summary"
        assert messages[2].tool_calls == calls
        assert [m.tool_call_id for m in messages if m.role == "tool"] == (
            ["native"] if partial else []
        )
        assert agent.llm.call_count == 0
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("cached", [False, True])
async def test_legacy_watermark_free_snapshot_and_valid_origin_resume(
    tmp_path, declared, cached
):
    store = SessionStore(str(tmp_path / "snapshot.kohakutr"))
    agent = await Agent.build(
        AgentConfig(name="fresh"), llm=ScriptedLLM(["continued"]), io="headless"
    )
    try:

        def seed():
            store.append_event(
                "alice",
                "user_message",
                {"content": "question"},
                turn_index=1,
                branch_id=1,
            )
            generation = {
                "schema": 1,
                "generation_id": "valid",
                "source": {"model": "historical"},
            }
            if declared:
                store.append_event(
                    "alice",
                    "assistant_generation_start",
                    generation.copy(),
                    turn_index=1,
                    branch_id=1,
                )
            store.append_event(
                "alice",
                "text_chunk",
                {
                    "content": "answer",
                    **({"generation_id": "valid"} if declared else {}),
                },
                turn_index=1,
                branch_id=1,
            )
            if not cached:
                return
            store.save_conversation(
                "alice",
                [
                    {
                        "role": "user",
                        "content": "question",
                        "metadata": {"turn_index": 1, "branch_id": 1},
                    },
                    {
                        "role": "assistant",
                        "content": "answer",
                        **(
                            {"metadata": {"generation": generation}} if declared else {}
                        ),
                    },
                ],
            )

        await store.run(seed)
        await inject_saved_state_async(agent, store, "alice")
        assert agent.config.name == "alice"
        assert [m.content for m in agent.controller.conversation.get_messages()] == [
            "question",
            "answer",
        ]
        await store.run(agent.attach_session_store, store)
        await agent.start()
        assert (await agent.run("next", timeout=5)).text == "continued"
        assert agent._parent_branch_path == [(1, 1)]
    finally:
        await agent.stop()
        await asyncio.to_thread(store.close)


@pytest.mark.parametrize(
    "corruption",
    [
        "source",
        "missing",
        "sibling",
        "invalid_schema",
        "unknown_source_field",
        "unknown_generation_field",
    ],
)
async def test_snapshot_generation_claim_requires_selected_origin(tmp_path, corruption):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original", "sibling"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question", timeout=5)
        await agent.regenerate_last_response()
        await agent.select_branch({1: 1})
        path = agent.session_store.path
    store = SessionStore(path)
    restored = await Agent.build(
        AgentConfig(name="fresh"), llm=ScriptedLLM(["must not run"]), io="headless"
    )
    try:

        def corrupt():
            snapshot = deepcopy(store.load_conversation("alice"))
            assistant = next(m for m in snapshot if m["role"] == "assistant")
            generation = assistant["metadata"]["generation"]
            match corruption:
                case "source":
                    generation["source"]["model"] = "forged-model"
                case "unknown_source_field":
                    generation["source"]["provider_config"] = "must reject"
                case "unknown_generation_field":
                    generation["unrecognized_authority"] = "must reject"
                case "missing":
                    generation["generation_id"] = "unknown-origin"
                case "invalid_schema":
                    generation["schema"] = True
                case "sibling":
                    origin = next(
                        e
                        for e in store.get_events("alice")
                        if e["type"] == "assistant_generation_start"
                        and e["branch_id"] == 2
                    )
                    assistant["metadata"]["generation"] = {
                        k: origin[k] for k in ("schema", "generation_id", "source")
                    }
            store.save_conversation("alice", snapshot)

        await store.run(corrupt)
        conversation = restored.controller.conversation
        messages = conversation.get_messages()[:]
        coords = (
            restored._turn_index,
            restored._branch_id,
            list(restored._parent_branch_path),
        )
        with pytest.raises(HistoryCompatibilityError):
            await inject_saved_state_async(restored, store, "alice")
        assert restored.controller.conversation is conversation
        assert conversation.get_messages() == messages
        assert (
            restored._turn_index,
            restored._branch_id,
            restored._parent_branch_path,
        ) == coords
        assert restored.config.name == "fresh"
        assert restored.llm.call_count == 0
    finally:
        await restored.stop()
        await asyncio.to_thread(store.close)
