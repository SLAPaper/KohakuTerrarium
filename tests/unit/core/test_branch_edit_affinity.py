"""Detached edit requests and canonical identities at rerun admission."""

import asyncio
import json
import threading

import pytest

import kohakuterrarium.core.branch_rerun as rerun

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.session.raw_history import UserMessageSelector
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize(
    "selector", ["message_idx", "user_position", "turn_index", "raw", "view"]
)
@pytest.mark.parametrize("legacy", [False, True])
async def test_edit_after_compact_keeps_canonical_turn_on_branch_round_trip(
    tmp_path, selector, legacy
):
    provider = ScriptedLLM(
        ["one", "two", "three", "edited", "regenerated", "continued"]
    )
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(
                name="alice",
                compact={
                    "max_tokens": 100000,
                    "keep_recent_turns": 1,
                    "cooldown_seconds": 0,
                },
            ),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("first")
        await creature.run("second")
        await creature.run("third")
        if legacy:
            await agent._submit_session_state(
                lambda: next(
                    m
                    for m in agent.controller.conversation.get_messages()
                    if m.role == "assistant"
                ).metadata.pop("generation")
            )
        agent.compact_manager._llm = ScriptedLLM(["COMPACTED_PREFIX"])
        assert agent.compact_manager.trigger_compact()
        await agent.compact_manager.wait_for_current()
        messages = agent.controller.conversation.get_messages()
        index = next(
            i
            for i, m in enumerate(messages)
            if m.role == "user" and m.content == "third"
        )
        assert messages[index].metadata["turn_index"] == 3
        options = (
            {"user_position": 0}
            if selector == "user_position"
            else ({"turn_index": 3} if selector == "turn_index" else {})
        )
        if selector == "raw":
            options = {
                "target": UserMessageSelector(
                    messages[index].metadata["event_id"], 3, 1
                )
            }
        elif selector == "view":
            options = {"turn_index": 3, "branch_view": {1: 1, 2: 1, 3: 1}}
        assert await agent.edit_and_rerun(index, "EDITED_THIRD", **options)
        assert agent._turn_index == 3
        assert "COMPACTED_PREFIX" in str(provider.last_messages)
        await agent.regenerate_last_response()
        assert agent._turn_index == 3
        view = dict(agent._parent_branch_path)
        view[3] = agent._branch_id
        await agent.select_branch({1: 1, 2: 1, 3: 1})
        assert "EDITED_THIRD" not in agent.controller.conversation.to_json()
        await agent.select_branch(view)
        await creature.run("after switch")
        request = [m for m in provider.last_messages if m["role"] != "system"]
        assert "COMPACTED_PREFIX" in str(request)
        assert [m["content"] for m in request if m["role"] == "user"] == [
            "EDITED_THIRD",
            "after switch",
        ]
        events = await agent.session_store.run(agent.session_store.get_events, "alice")
        edits = [
            e
            for e in events
            if e["type"] == "user_message" and e.get("content") == "EDITED_THIRD"
        ]
        assert edits and all(e["turn_index"] == 3 for e in edits)


@pytest.mark.parametrize("check", ["metadata", "system"])
async def test_raw_edit_retains_system_and_installs_canonical_user(tmp_path, check):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original", "edited answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        system = agent.controller.conversation.get_system_message()
        config = agent.controller.conversation.config
        events = await agent.session_store.run(agent.session_store.get_events, "alice")
        target = next(e for e in events if e["type"] == "user_message")
        assert await agent.edit_and_rerun(
            -1, "edited", target=UserMessageSelector(target["event_id"], 1, 1)
        )
        assert agent.controller.conversation.config is config
        if check == "system":
            assert agent.controller.conversation.get_system_message() is system
        else:
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            written = next(
                e for e in events if e["type"] == "user_message" and e["branch_id"] == 2
            )
            user = next(
                m
                for m in agent.controller.conversation.get_messages()
                if m.role == "user"
            )
            assert user.metadata["event_id"] == written["event_id"]
            assert user.metadata["turn_index"] == 1
            assert user.metadata["branch_id"] == 2
            assert user.metadata["event_origin"][-1] == written["event_id"]
            assert user.metadata["user_pair_id"] == written["user_pair_id"]
            first_origin = user.metadata["event_origin"]
            await agent.regenerate_last_response()
            regenerated = next(
                m
                for m in agent.controller.conversation.get_messages()
                if m.role == "user"
            )
            assert regenerated.metadata["event_origin"] != first_origin
            assert (
                regenerated.metadata["event_origin"][-1]
                == regenerated.metadata["event_id"]
            )
            assert (
                regenerated.metadata["user_pair_origin"][-1]
                < regenerated.metadata["event_id"]
            )


@pytest.mark.parametrize("target_kind", ["raw", "view"])
async def test_edit_rejects_duplicate_dispatch_before_replay_deduplication(
    tmp_path, target_kind
):
    provider = ScriptedLLM(["one", "two", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("first", timeout=5)).text == "one"
        store = agent.session_store

        def append_invalid_pair():
            call = {
                "id": "native-call",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }
            store.append_event(
                "alice",
                "assistant_tool_calls",
                {"tool_calls": [call]},
                turn_index=1,
                branch_id=1,
            )
            for _ in range(2):
                store.append_event(
                    "alice",
                    "tool_call",
                    {
                        "call_id": "dispatch",
                        "tool_call_id": "native-call",
                        "name": "read",
                        "args": {},
                    },
                    turn_index=1,
                    branch_id=1,
                )
            store.append_event(
                "alice",
                "tool_result",
                {
                    "call_id": "dispatch",
                    "tool_call_id": "native-call",
                    "name": "read",
                    "output": "done",
                },
                turn_index=1,
                branch_id=1,
            )

        await store.run(append_invalid_pair)
        assert (await creature.run("second", timeout=5)).text == "two"
        before = agent.controller.conversation
        raw = await store.run(store.get_events, "alice")
        target = next(
            e for e in raw if e["type"] == "user_message" and e["turn_index"] == 2
        )
        with pytest.raises(ValueError, match="duplicate dispatch"):
            if target_kind == "raw":
                await agent.edit_and_rerun(
                    -1, "edited", target=UserMessageSelector(target["event_id"], 2, 1)
                )
            else:
                await agent.edit_and_rerun(
                    -1, "edited", turn_index=2, branch_view={1: 1, 2: 1}
                )
        assert agent.controller.conversation is before
        assert await store.run(store.get_events, "alice") == raw
        assert provider.call_count == 2


def _child(kind, *, job, provider=None, name="child", task="work", output="done"):
    common = {
        "job_id": job,
        "name": name,
        "task": task,
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
        "origin": {
            "runtime_scope_id": "scope",
            "delivery_id": f"scope:{job}",
            "origin_path": [[1, 1]],
            "turn_index": 1,
            "branch_id": 1,
            "parent_branch_path": [],
        },
    }
    if provider:
        common["tool_call_id"] = provider
        common["tool_call_arguments"] = json.dumps(
            {"task": task, "run_in_background": False}
        )
    if kind == "call":
        return {"type": "subagent_call", **common}
    return {"type": "subagent_result", "output": output, **common}


def _append_children(store, events):
    for event in events:
        payload = {
            key: value
            for key, value in event.items()
            if key
            not in {"type", "turn_index", "branch_id", "parent_branch_path", "event_id"}
        }
        store.append_event(
            "alice",
            event["type"],
            payload,
            turn_index=event.get("turn_index"),
            branch_id=event.get("branch_id"),
            parent_branch_path=event.get("parent_branch_path"),
        )


async def test_tail_regeneration_uses_user_input_not_replayed_child_feedback(tmp_path):
    provider = ScriptedLLM(["original", "regenerated"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("real question")
        await agent.session_store.run(
            _append_children,
            agent.session_store,
            [
                _child("call", job="agent_child_kept"),
                _child("result", job="agent_child_kept"),
            ],
        )
        await agent.select_branch(None)
        assert "agent_child_kept" in agent.controller.conversation.to_json()
        await agent.regenerate_last_response()
        events = await agent.session_store.run(agent.session_store.get_events, "alice")
        new_users = [
            e for e in events if e["type"] == "user_message" and e["branch_id"] == 2
        ]
        assert [e["content"] for e in new_users] == ["real question"]
        assert "agent_child_kept" not in agent.controller.conversation.to_json()
        assert "regenerated" in agent.controller.conversation.to_json()


@pytest.mark.parametrize("locator", ["raw", "index", "branch_view"])
@pytest.mark.parametrize("native", [False, True])
async def test_edit_prefix_keeps_completed_child_and_ignores_discarded_tail(
    tmp_path, locator, native
):
    provider = ScriptedLLM(["one", "two", "edited answer"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "one"
        store = agent.session_store
        provider_id = "provider-child" if native else None
        retained = [
            _child("call", job="agent_child_kept", provider=provider_id),
            _child("result", job="agent_child_kept", provider=provider_id),
        ]
        discarded = [
            _child(
                "call",
                job="agent_child_tail",
                provider="provider-tail" if native else None,
                task="later",
            )
        ]
        for event in discarded:
            event["turn_index"] = 1
            event["branch_id"] = 2
            event["parent_branch_path"] = []
            event["origin"] = {
                **event["origin"],
                "origin_path": [[1, 2]],
                "branch_id": 2,
            }
        await store.run(_append_children, store, [*retained, *discarded])
        assert (await creature.run("next", timeout=5)).text == "two"
        discarded_target = _child(
            "call", job="agent_child_tail_target", provider="missing-args"
        )
        discarded_target.update(turn_index=2, branch_id=1, parent_branch_path=[(1, 1)])
        discarded_target.pop("tool_call_arguments")
        await store.run(_append_children, store, [discarded_target])
        raw = await store.run(store.get_events, "alice")
        target = next(
            event
            for event in raw
            if event["type"] == "user_message" and event["content"] == "next"
        )
        if locator == "raw":
            admitted = await agent.edit_and_rerun(
                -1,
                "edited",
                target=UserMessageSelector(target["event_id"], 2, 1),
            )
        elif locator == "index":
            index = next(
                i
                for i, message in enumerate(
                    agent.controller.conversation.get_messages()
                )
                if message.role == "user" and message.content == "next"
            )
            admitted = await agent.edit_and_rerun(index, "edited", turn_index=2)
        else:
            admitted = await agent.edit_and_rerun(
                -1, "edited", turn_index=2, branch_view={1: 1, 2: 1}
            )
        assert admitted is True
        messages = agent.controller.conversation.get_messages()
        users = [
            message
            for message in messages
            if message.role == "user" and message.metadata.get("kind") != "tool_results"
        ]
        assert [message.content for message in users] == ["question", "edited"]
        assert users[1].metadata["turn_index"] == 2
        assert users[1].metadata["branch_id"] == 2
        if native:
            announcements = [
                message
                for message in messages
                if message.tool_calls
                and message.tool_calls[0]["id"] == "provider-child"
            ]
            assert len(announcements) == 1
            announcement = announcements[0]
            assert (
                announcement.tool_calls[0]["function"]["arguments"]
                == retained[0]["tool_call_arguments"]
            )
            result = next(
                message
                for message in messages
                if message.role == "tool" and message.tool_call_id == "provider-child"
            )
            assert result.content == "done"
        else:
            feedbacks = [
                message
                for message in messages
                if message.role == "user" and "agent_child_kept" in str(message.content)
            ]
            assert len(feedbacks) == 1
            feedback = feedbacks[0]
            assert feedback.metadata["kind"] == "tool_results"
            assert "done" in feedback.content
            assert not any(
                getattr(message, "metadata", {}).get("turn_index") == 2
                and message is not users[1]
                for message in messages
            )
        assert not any(
            "agent_child_tail" in str(message.content) for message in messages
        )
        sent = provider.call_log[-1]
        assert any(
            item.get("role") == "user" and item.get("content") == "edited"
            for item in sent
        )
        if locator == "index":
            await creature.run("third", timeout=5)
            await creature.run("fourth", timeout=5)
            index = next(
                i
                for i, message in enumerate(
                    agent.controller.conversation.get_messages()
                )
                if message.role == "user" and message.content == "third"
            )
            assert await agent.edit_and_rerun(index, "third edited")
            assert agent._turn_index == 3
            assert agent._branch_id == 2
            users_after = [
                m.content
                for m in agent.controller.conversation.get_messages()
                if m.role == "user" and m.metadata.get("kind") != "tool_results"
            ]
            assert users_after == ["question", "edited", "third edited"]


@pytest.mark.parametrize(
    "damage", ["duplicate_provider", "conflicting_provider", "pending", "missing_args"]
)
@pytest.mark.parametrize("locator", ["raw", "index", "branch_view"])
async def test_retained_corrupt_child_rejects_before_persist(tmp_path, damage, locator):
    provider = ScriptedLLM(["one", "two", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("first", timeout=5)).text == "one"
        store = agent.session_store
        events = [
            _child("call", job="agent_child_one", provider="provider-child"),
            _child("result", job="agent_child_one", provider="provider-child"),
        ]
        if damage == "duplicate_provider":
            events.extend(
                [
                    _child("call", job="agent_child_two", provider="provider-child"),
                    _child("result", job="agent_child_two", provider="provider-child"),
                ]
            )
        elif damage == "conflicting_provider":
            events[1]["tool_call_id"] = "provider-other"
        elif damage == "pending":
            events.pop()
        else:
            events[0].pop("tool_call_arguments")
        await store.run(_append_children, store, events)
        assert (await creature.run("second", timeout=5)).text == "two"
        before = agent.controller.conversation
        raw = await store.run(store.get_events, "alice")
        target = next(
            event
            for event in raw
            if event["type"] == "user_message" and event["content"] == "second"
        )
        with pytest.raises(ValueError):
            if locator == "raw":
                await agent.edit_and_rerun(
                    -1,
                    "edited",
                    target=UserMessageSelector(
                        target["event_id"], target["turn_index"], target["branch_id"]
                    ),
                )
            else:
                await agent.edit_and_rerun(
                    -1,
                    "edited",
                    turn_index=2,
                    branch_view={1: 1, 2: 1} if locator == "branch_view" else None,
                )
        assert agent.controller.conversation is before
        assert await store.run(store.get_events, "alice") == raw
        assert provider.call_count == 2


async def test_new_multimodal_edit_is_not_treated_as_legacy_switch_evidence(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original", "edited answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        await agent.session_store.run(
            agent.session_store.append_event,
            "alice",
            "model_switch",
            {
                "schema": 1,
                "classification": "portable",
                "from_source": {"backend_key": "old", "backend_type": "openai"},
                "to_source": {"backend_key": "new", "backend_type": "openai"},
            },
        )
        content = [{"type": "text", "text": "edited question"}]
        assert await agent.edit_and_rerun(-1, content, turn_index=1)
        messages = agent.controller.conversation.get_messages()
        assert [m.content for m in messages if m.role == "assistant"] == [
            "edited answer"
        ]
        events = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert (
            next(
                e["content"]
                for e in events
                if e["type"] == "user_message" and e["branch_id"] == 2
            )
            == content
        )


async def test_edit_captures_request_before_worker_wait(tmp_path, monkeypatch):
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    prepare = rerun.prepare_edit

    def held(*args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return prepare(*args)

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original", "edited answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        view = {1: 1}
        monkeypatch.setattr(rerun, "prepare_edit", held)
        task = asyncio.create_task(
            agent.edit_and_rerun(-1, "edited", turn_index=1, branch_view=view)
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            view[1] = 999
            release.set()
            assert await asyncio.wait_for(task, 5)
            assert agent._branch_id == 2
            assert [
                m.content
                for m in agent.controller.conversation.get_messages()
                if m.role == "user"
            ] == ["edited"]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
