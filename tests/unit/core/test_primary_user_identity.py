"""Ordinary user identities are tied to accepted writes and one admitted turn."""

import asyncio
import threading
from copy import copy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core import agent_event_loop
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.event_inbox import EventEnvelope
from kohakuterrarium.core.events import create_user_input_event
from kohakuterrarium.llm.message import TextPart
from kohakuterrarium.session.raw_history import (
    UserMessageSelector,
    select_raw_history_prefix,
)
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.modules.plugin.base import BasePlugin


async def test_agent_only_legacy_store_records_without_canonical_identity(tmp_path):
    store = SessionStore(tmp_path / "legacy.kohakutr")
    try:
        async with Terrarium() as engine:
            creature = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM(["legacy answer"]),
                io="headless",
                start=True,
            )
            agent = creature.agent
            assert agent.controller.session_store is None
            agent.session_store = store
            assert (await creature.run("question")).text == "legacy answer"
            live = [
                m
                for m in agent.controller.conversation.get_messages()
                if m.role == "user"
            ][-1]
            assert "event_id" not in live.metadata
            rows = await store.run(store.get_events, "alice")
            assert [e["content"] for e in rows if e["type"] == "user_message"] == [
                "question"
            ]
    finally:
        store.close(update_status=False)


async def test_subscriber_cannot_mutate_receipt_or_next_user_pair_row(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        receipts = []
        capture = agent_event_loop.capture_primary_user

        def observe(*args):
            receipt = capture(*args)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(agent_event_loop, "capture_primary_user", observe)

        def mutate(_key, row):
            if row["type"] == "user_input":
                row["content"][0]["text"] = "subscriber mutation"

        await agent.session_store.run(agent.session_store.subscribe, mutate)
        event = create_user_input_event([TextPart(text="original")])
        await agent._run_admitted_batch([EventEnvelope(event)])
        assert receipts[0].payload["content"] == [{"type": "text", "text": "original"}]
        rows = await agent.session_store.run(agent.session_store.get_events, "alice")
        user = [e for e in rows if e["type"] == "user_message"][-1]
        assert user["content"] == [{"type": "text", "text": "original"}]
        live = [
            m for m in agent.controller.conversation.get_messages() if m.role == "user"
        ][-1]
        assert live.metadata["event_id"] == user["event_id"]
        assert event.content[0].text == "original"


async def test_ordinary_user_identity_selects_persisted_event(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first", "second"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        for answer in ("first", "second"):
            assert (await creature.run("same question")).text == answer
            live = [
                m
                for m in agent.controller.conversation.get_messages()
                if m.role == "user"
            ][-1]
            raw = await agent.session_store.run(agent.session_store.get_events, "alice")
            expected = [e for e in raw if e["type"] == "user_message"][-1]
            identity = {
                key: expected[key] for key in ("event_id", "turn_index", "branch_id")
            }
            assert {key: live.metadata.get(key) for key in identity} == identity
            prefix = select_raw_history_prefix(
                raw, selector=UserMessageSelector(**identity)
            )
            assert prefix.target["content"] == "same question"
            assert prefix.target["event_id"] == expected["event_id"]
            assert getattr(agent, "_primary_user_receipt", None) is None


@pytest.mark.parametrize(
    "field",
    [
        "store",
        "controller",
        "controller_store",
        "controller_unbound",
        "conversation",
        "namespace",
        "branch",
        "path",
        "view",
    ],
)
@pytest.mark.parametrize("stage", ["write", "capture"])
async def test_changed_binding_rejects_user_receipt(
    tmp_path, monkeypatch, field, stage
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["must not run"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        owner, entered, release = (
            agent._session_state_owner,
            asyncio.Event(),
            threading.Event(),
        )
        loop = asyncio.get_running_loop()
        append, execute = agent_event_loop.append_user_event_pair, owner._execute
        held = []

        def held_append(*args, **kwargs):
            assert threading.current_thread() is not threading.main_thread()
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(8)
            return append(*args, **kwargs)

        def held_capture(callback, future):
            if "take_primary_user" in getattr(
                getattr(callback, "__code__", None), "co_names", ()
            ):
                held.append((callback, future))
                entered.set()
            else:
                execute(callback, future)

        if stage == "write":
            monkeypatch.setattr(agent_event_loop, "append_user_event_pair", held_append)
        else:
            monkeypatch.setattr(owner, "_execute", held_capture)
        replacement = SessionStore(tmp_path / "replacement.kohakutr")
        original_controller = agent.controller
        target, attribute, value = {
            "store": (agent, "session_store", replacement),
            "controller": (agent, "controller", copy(original_controller)),
            "controller_store": (original_controller, "session_store", replacement),
            "controller_unbound": (original_controller, "session_store", None),
            "conversation": (original_controller, "conversation", Conversation()),
            "namespace": (agent.config, "name", "other"),
            "branch": (agent, "_branch_id", 99),
            "path": (agent, "_parent_branch_path", [(99, 99)]),
            "view": (agent, "_branch_view", {99: 99}),
        }[field]
        pending = asyncio.create_task(creature.run("identity", raise_on_error=False))
        missing, changed = object(), False
        old = missing
        try:
            await asyncio.wait_for(entered.wait(), 8)
            old = getattr(target, attribute, missing)
            changed = True
            await owner.submit_internal(lambda: setattr(target, attribute, value))
            release.set()
            if held:
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.wait_for(pending, 8)
            assert llm.call_count == 0
            assert getattr(agent, "_primary_user_receipt", None) is None
        finally:
            release.set()
            for callback, future in held:
                if not future.done():
                    await owner.submit_internal(lambda: execute(callback, future))
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            if changed:
                await owner.submit_internal(
                    lambda: (
                        delattr(target, attribute)
                        if old is missing
                        else setattr(target, attribute, old)
                    )
                )
            replacement.close(update_status=False)


@pytest.mark.parametrize("mode", ["text", "multimodal", "forged_flags"])
async def test_plugin_changed_input_runs_without_false_identity(tmp_path, mode):
    class Mutate(BasePlugin):
        name = "mutate-user"

        async def on_event(self, event, **kwargs):
            if event.type != "user_input":
                return
            if mode == "multimodal":
                event.content[0].text = "changed"
            elif mode == "text":
                event.content = "changed"
            else:
                event.context.update(
                    rerun=True,
                    edited=True,
                    _history_user_identity={
                        "event_id": 999,
                        "turn_index": 999,
                        "branch_id": 999,
                    },
                )

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            plugins=[Mutate()],
            io="headless",
            start=True,
        )
        event = create_user_input_event(
            [TextPart(text="original")] if mode == "multimodal" else "original"
        )
        await creature.agent._run_admitted_batch([EventEnvelope(event)])
        agent = creature.agent
        assert agent.llm.call_count == 1
        users = [
            m for m in agent.controller.conversation.get_messages() if m.role == "user"
        ]
        assert "event_id" not in users[-1].metadata
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        original = [e for e in raw if e["type"] == "user_message"][-1]["content"]
        assert original == (
            [{"type": "text", "text": "original"}]
            if mode == "multimodal"
            else "original"
        )
        assert getattr(agent, "_primary_user_receipt", None) is None


@pytest.mark.parametrize("batching", [False, True])
async def test_reused_event_and_compound_admission_do_not_reuse_identity(
    tmp_path, batching
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["one", "two", "three", "four"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        event = create_user_input_event("same")
        identities = []
        for _ in range(2):
            await agent._run_admitted_batch([EventEnvelope(event)])
            identities.append(
                [
                    m
                    for m in agent.controller.conversation.get_messages()
                    if m.role == "user"
                ][-1].metadata["event_id"]
            )
        assert identities[0] != identities[1]
        agent.controller.config.batch_stackable_events = batching
        before = len(
            [
                m
                for m in agent.controller.conversation.get_messages()
                if m.role == "user"
            ]
        )
        await agent._run_admitted_batch(
            [EventEnvelope(event), EventEnvelope(create_user_input_event("folded"))]
        )
        if not batching:
            assert agent.controller.has_pending_events()
            async for _ in agent.controller.run_once():
                pass
        users = [
            m for m in agent.controller.conversation.get_messages() if m.role == "user"
        ]
        assert len(users) > before
        assert all("event_id" not in message.metadata for message in users[before:])
        assert getattr(agent, "_primary_user_receipt", None) is None
        assert "_history_user_identity" not in event.context


@pytest.mark.parametrize("queue", ["pending", "queue"])
async def test_older_controller_occurrence_cannot_consume_new_identity(tmp_path, queue):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["old", "new"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        agent.controller.config.batch_stackable_events = False
        event = create_user_input_event("same object")
        if queue == "pending":
            agent.controller._pending_events.append(event)
        else:
            await agent.controller.push_event(event)
        await agent._run_admitted_batch([EventEnvelope(event)])
        assert agent.controller.has_pending_events()
        assert getattr(agent, "_primary_user_receipt", None) is None
        async for _ in agent.controller.run_once():
            pass
        users = [
            m for m in agent.controller.conversation.get_messages() if m.role == "user"
        ]
        assert len(users) == 2
        assert all("event_id" not in message.metadata for message in users)
        assert getattr(agent, "_primary_user_receipt", None) is None


@pytest.mark.parametrize("failure", ["write", "output", "push"])
async def test_failed_primary_turn_does_not_leak_receipt(
    tmp_path, monkeypatch, failure
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["next"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        event = create_user_input_event("failed")
        with monkeypatch.context() as patch:
            if failure == "write":

                def fail_write(*args, **kwargs):
                    raise OSError("write failed")

                patch.setattr(agent_event_loop, "append_user_event_pair", fail_write)
            elif failure == "output":
                emit = agent.output_router.emit

                async def fail_output(output):
                    if output.type == "user_input":
                        raise OSError("output failed")
                    return await emit(output)

                patch.setattr(agent.output_router, "emit", fail_output)
            else:

                async def fail_push(event):
                    raise OSError("push failed")

                patch.setattr(agent.controller, "push_event", fail_push)
            with pytest.raises(OSError, match="failed"):
                await agent._run_admitted_batch([EventEnvelope(event)])
        assert getattr(agent, "_primary_user_receipt", None) is None
        assert agent._active_event_run is None
        assert agent._turn_completion is None
        assert (await creature.run("next question")).text == "next"
        live = [
            m for m in agent.controller.conversation.get_messages() if m.role == "user"
        ][-1]
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert (
            live.metadata["event_id"]
            == [e["event_id"] for e in raw if e["type"] == "user_message"][-1]
        )


@pytest.mark.parametrize(
    "operation", ["capture_primary_user", "install_primary_user", "arm_primary_user"]
)
async def test_cancelled_owner_receipt_phase_does_not_start_provider(
    tmp_path, monkeypatch, operation
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["must not run"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        owner, held, entered = agent._session_state_owner, [], asyncio.Event()
        execute = owner._execute

        def hold(callback, future):
            if operation in getattr(
                getattr(callback, "__code__", None), "co_names", ()
            ):
                held.append((callback, future))
                entered.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        pending = asyncio.create_task(
            agent._run_admitted_batch(
                [EventEnvelope(create_user_input_event("question"))]
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        pending.cancel()
        await owner.submit_internal(lambda: execute(*held[0]))
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert llm.call_count == 0
        assert agent._active_event_run is None
        assert getattr(agent, "_primary_user_receipt", None) is None
        rows = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert len([row for row in rows if row["type"] == "user_message"]) == (
            0 if operation == "capture_primary_user" else 1
        )


async def test_repeated_cancel_during_receipt_cleanup_still_clears_turn(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        owner, held, entered = agent._session_state_owner, [], asyncio.Event()
        execute = owner._execute

        def hold(callback, future):
            if "clear_primary_user" in getattr(
                getattr(callback, "__code__", None), "co_names", ()
            ):
                held.append((callback, future))
                entered.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        pending = asyncio.create_task(
            agent._run_admitted_batch(
                [EventEnvelope(create_user_input_event("question"))]
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        scope = agent._turn_completion
        pending.cancel()
        await asyncio.sleep(0)
        pending.cancel()
        await owner.submit_internal(lambda: execute(*held[0]))
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert agent._active_event_run is None
        assert agent._turn_completion is None
        assert scope.done()
        assert getattr(agent, "_primary_user_receipt", None) is None


async def test_cancelled_primary_write_joins_accepted_pair(tmp_path, monkeypatch):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["unused"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        append = agent_event_loop.append_user_event_pair

        def held(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            return append(*args, **kwargs)

        monkeypatch.setattr(agent_event_loop, "append_user_event_pair", held)
        pending = asyncio.create_task(
            agent._record_primary_user_input(create_user_input_event("accepted"))
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            pending.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not pending.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [
            (e["type"], e["content"])
            for e in raw
            if e["type"] in ("user_input", "user_message")
        ] == [("user_input", "accepted"), ("user_message", "accepted")]
        assert getattr(agent, "_primary_user_receipt", None) is None
