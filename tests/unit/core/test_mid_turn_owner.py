"""Mid-turn input and background folds retain owner and feedback identity."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.conversation_elide import TOOL_FEEDBACK_KIND
from kohakuterrarium.core.event_inbox import EventEnvelope
from kohakuterrarium.core.events import (
    create_tool_complete_event,
    create_user_input_event,
)
from kohakuterrarium.core.message_locator import is_turn_user_message
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("background", [False, True])
async def test_mid_turn_append_and_debt_commit_on_owner(
    tmp_path, monkeypatch, background
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        event = (
            create_tool_complete_event(
                job_id="work_job",
                content="folded result",
                **capture_tool_context(agent, "work_job", "background", None),
            )
            if background
            else create_user_input_event("folded question")
        )
        agent._event_inbox.put(EventEnvelope(event=event))
        agent._turn_dispatched_bg.add("work_job")
        owner = agent._session_state_owner.task
        writes, observers, removals = [], [], []

        class ObservedSet(set):
            def discard(self, item):
                removals.append(asyncio.current_task())
                return super().discard(item)

        agent._turn_dispatched_bg = ObservedSet(agent._turn_dispatched_bg)
        conversation = agent.controller.conversation
        append, notify = type(conversation).append, agent.output_router.notify_activity

        def observe_append(self, role, content, **kwargs):
            if self is conversation:
                writes.append(asyncio.current_task())
            return append(self, role, content, **kwargs)

        def observe_activity(*args, **kwargs):
            observers.append(asyncio.current_task())
            return notify(*args, **kwargs)

        monkeypatch.setattr(type(conversation), "append", observe_append)
        monkeypatch.setattr(agent.output_router, "notify_activity", observe_activity)
        async with agent._processing_lock:
            assert await agent._drain_mid_turn_pending_inputs(agent.controller) == 1
        assert writes == [owner]
        assert all(task is not owner for task in observers)
        if background:
            assert removals == [owner]
        else:
            assert not removals
        assert agent.llm.call_count == 1


@pytest.mark.parametrize("include_user", [False, True])
async def test_background_fold_does_not_become_editable_user_turn(
    tmp_path, include_user
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        event = create_tool_complete_event(
            job_id="work_job",
            content="folded result",
            **capture_tool_context(agent, "work_job", "background", None),
        )
        agent._event_inbox.put(EventEnvelope(event=event))
        if include_user:
            agent._event_inbox.put(
                EventEnvelope(event=create_user_input_event("real followup"))
            )
        async with agent._processing_lock:
            assert (
                await agent._drain_mid_turn_pending_inputs(agent.controller)
                == 1 + include_user
            )
        message = agent.controller.conversation.get_messages()[-1]
        assert is_turn_user_message(message) is include_user
        assert (message.metadata.get("kind") == TOOL_FEEDBACK_KIND) is (
            not include_user
        )


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("after_append", [False, True, "replaced"])
async def test_failed_fold_reports_error_without_delivery_or_partial_append(
    tmp_path, monkeypatch, background, after_append
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        event = (
            create_tool_complete_event(
                job_id="work_job",
                content="folded result",
                **capture_tool_context(agent, "work_job", "background", None),
            )
            if background
            else create_user_input_event("folded question")
        )
        future = asyncio.get_running_loop().create_future()
        agent._event_inbox.put(EventEnvelope(event=event, future=future))
        agent._turn_dispatched_bg.add("work_job")
        conversation = agent.controller.conversation
        before = conversation.to_json()
        append = type(conversation).append

        def fail(self, *args, **kwargs):
            if self is conversation:
                if after_append:
                    append(self, *args, **kwargs)
                raise ValueError("fold commit failed")
            return append(self, *args, **kwargs)

        if after_append == "replaced":
            execute = agent._session_state_owner._execute

            def replace_before_commit(callback, future):
                if "_fold_mid_turn_inputs.<locals>.commit" in callback.__qualname__:
                    agent.controller.conversation = Conversation.from_json(before)
                execute(callback, future)

            monkeypatch.setattr(
                agent._session_state_owner, "_execute", replace_before_commit
            )
        else:
            monkeypatch.setattr(type(conversation), "append", fail)
        async with agent._processing_lock:
            with pytest.raises(
                RuntimeError if after_append == "replaced" else ValueError
            ):
                await agent._drain_mid_turn_pending_inputs(agent.controller)
        assert conversation.to_json() == before
        assert agent.controller.conversation.to_json() == before
        assert future.done() and future.result().status == "error"
        assert "work_job" in agent._turn_dispatched_bg
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert not [
            event
            for event in raw
            if event["type"] in ("user_input_injected", "background_result")
        ]
        assert agent.llm.call_count == 1


async def test_cancelled_fold_joins_all_claimed_input_records(tmp_path, monkeypatch):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        for content in ("first folded", "second folded"):
            agent._event_inbox.put(
                EventEnvelope(event=create_user_input_event(content))
            )
        entered, release = asyncio.Event(), threading.Event()
        store, loop = agent.session_store, asyncio.get_running_loop()
        append = type(store).append_event

        def held(self, namespace, kind, data, **kwargs):
            if (
                self is store
                and kind == "user_input_injected"
                and data["content"] == "first folded"
            ):
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
            return append(self, namespace, kind, data, **kwargs)

        monkeypatch.setattr(type(store), "append_event", held)
        async with agent._processing_lock:
            fold = asyncio.create_task(
                agent._drain_mid_turn_pending_inputs(agent.controller)
            )
            try:
                await asyncio.wait_for(entered.wait(), 3)
                fold.cancel()
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(fold), 0.05)
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await fold
                raw = await store.run(store.get_events, "alice")
                assert [
                    e["content"] for e in raw if e["type"] == "user_input_injected"
                ] == ["first folded", "second folded"]
                assert (
                    agent.controller.conversation.get_messages()[-1].content
                    == "first folded\n\nsecond folded"
                )
            finally:
                release.set()
                await asyncio.gather(fold, return_exceptions=True)


async def test_fold_copies_content_before_owner_wait_and_records_captured_coordinates(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        event = create_user_input_event([{"type": "text", "text": "original"}])
        agent._event_inbox.put(EventEnvelope(event=event))
        ready, saved = asyncio.Event(), []
        owner, execute = agent._session_state_owner, agent._session_state_owner._execute

        def intercept(callback, future):
            if (
                "_fold_mid_turn_inputs.<locals>.commit" in callback.__qualname__
                and not saved
            ):
                saved.append((callback, future))
                ready.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", intercept)
        async with agent._processing_lock:
            fold = asyncio.create_task(
                agent._drain_mid_turn_pending_inputs(agent.controller)
            )
            try:
                await asyncio.wait_for(ready.wait(), 3)
                event.content[0].text = "mutated"
                await owner.submit_internal(lambda: execute(*saved[0]))
                assert await fold == 1
            finally:
                if saved and not saved[0][1].done():
                    await owner.submit_internal(lambda: execute(*saved[0]))
                await asyncio.gather(fold, return_exceptions=True)
        message = agent.controller.conversation.get_messages()[-1]
        assert message.get_text_content() == "original"
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        injected = [e for e in raw if e["type"] == "user_input_injected"]
        assert len(injected) == 1 and injected[0]["content"][0]["text"] == "original"
        assert (injected[0]["turn_index"], injected[0]["branch_id"]) == (1, 1)


async def test_stop_joins_owner_accepted_midturn_input_without_new_llm(
    tmp_path, monkeypatch
):
    entered, release = asyncio.Event(), asyncio.Event()

    class Provider(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            entered.set()
            yield "first"
            await release.wait()

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = Provider([])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        ready, held = asyncio.Event(), []
        owner, execute = agent._session_state_owner, agent._session_state_owner._execute

        def intercept(callback, future):
            if (
                "_fold_mid_turn_inputs.<locals>.commit" in callback.__qualname__
                and not held
            ):
                held.append((callback, future))
                ready.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", intercept)
        turn = asyncio.create_task(creature.run("question", raise_on_error=False))
        stopped = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert await agent.inject_input("accepted during turn") is False
            release.set()
            await asyncio.wait_for(ready.wait(), 3)
            stopped = asyncio.create_task(agent.stop())
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(stopped), 0.05)
            await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.wait_for(stopped, 3)
            await turn
            assert llm.call_count == 1
            assert agent._stopped
            raw = await agent.session_store.run(agent.session_store.get_events, "alice")
            assert [
                e["content"] for e in raw if e["type"] == "user_input_injected"
            ] == ["accepted during turn"]
        finally:
            release.set()
            if held and not held[0][1].done():
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.gather(
                *[task for task in (turn, stopped) if task is not None],
                return_exceptions=True,
            )
