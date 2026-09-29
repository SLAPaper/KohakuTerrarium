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
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.terrarium import session_coord
from kohakuterrarium.session.compact_compile import compile_coverage
from kohakuterrarium.session.reader import SessionReader
from kohakuterrarium.session.resume_replay import replay_resume_history


@pytest.mark.parametrize("batch", [False, True])
async def test_pure_user_group_retains_distinct_occurrences(tmp_path, batch):
    entered, release, responded = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Parent(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            if not batch and self.call_count == 0:
                entered.set()
                await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk
            responded.set()

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=Parent(["initial", "received"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        turn = None
        try:
            if batch:
                agent.pause()
            else:
                turn = asyncio.create_task(creature.run("start fold"))
                await asyncio.wait_for(entered.wait(), 3)
            for _ in range(2):
                assert await agent.inject_input("same user content") is False
            if batch:
                agent.resume()
                await asyncio.wait_for(responded.wait(), 3)
            else:
                release.set()
                assert (await asyncio.wait_for(turn, 5)).ok
            async with agent._processing_lock:
                group = agent.controller.conversation.get_messages()
                group = next(
                    message
                    for message in group
                    if message.role == "user"
                    and "same user content" in str(message.content)
                )
                assert group.metadata.get("kind") == "input_group"
                assert group.metadata.get("feedback_admission_origin")
                assert not any(
                    key in group.metadata
                    for key in ("event_id", "user_pair_id", "user_pair_origin")
                )
                store = agent.session_store
                coverage = await store.run(
                    compile_coverage,
                    store,
                    "alice",
                    [{**group.to_dict(), "metadata": group.metadata}],
                    [],
                )
                assert coverage["coverage_mode"] == "exact"
                path = store.path
        finally:
            release.set()
            if turn is not None:
                await asyncio.gather(turn, return_exceptions=True)
    with SessionReader(path) as reader:
        restored = replay_resume_history(list(reader.events("alice")))
        groups = [
            message
            for message in restored
            if message.get("metadata", {}).get("kind") == "input_group"
        ]
        assert (
            len(groups) == 1
            and groups[0]["content"] == "same user content\n\nsame user content"
        )


async def test_feedback_fold_during_copy_rebinds_without_joining_turn_lock(
    tmp_path, monkeypatch
):
    (
        tool_release,
        generating,
        provider_release,
        queued,
        folding,
        fold_release,
        copying,
    ) = [asyncio.Event() for _ in range(7)]
    copy_release = threading.Event()
    loop = asyncio.get_running_loop()

    class Work(BaseTool):
        tool_name = "work"
        description = "Held background work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            await tool_release.wait()
            return ToolResult(output="rebound feedback")

    class Parent(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            if self.call_count == 1:
                generating.set()
                await provider_release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=Parent(
                [
                    "[/work]@@run_in_background=true\n{}[work/]",
                    "independent",
                    "received",
                ]
            ),
            tools=[Work()],
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        agent, source = alice.agent, alice.agent.session_store
        await alice.run("start")
        put, fold, capture = (
            agent._event_inbox.put,
            agent._fold_mid_turn_inputs,
            session_coord.capture_runtime_transfer,
        )

        def observe_put(envelope):
            result = put(envelope)
            if envelope.event.type == "tool_complete":
                queued.set()
            return result

        async def held_fold(*args):
            folding.set()
            await fold_release.wait()
            return await fold(*args)

        first = True

        def held_copy(*args):
            nonlocal first
            result = capture(*args)
            if first and args[0] is source:
                first = False
                loop.call_soon_threadsafe(copying.set)
                assert copy_release.wait(8)
            return result

        monkeypatch.setattr(agent._event_inbox, "put", observe_put)
        monkeypatch.setattr(agent, "_fold_mid_turn_inputs", held_fold)
        monkeypatch.setattr(session_coord, "capture_runtime_transfer", held_copy)
        turn = asyncio.create_task(alice.run("independent work"))
        moving = None
        try:
            await asyncio.wait_for(generating.wait(), 3)
            tool_release.set()
            await asyncio.wait_for(queued.wait(), 3)
            provider_release.set()
            await asyncio.wait_for(folding.wait(), 3)
            moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
            await asyncio.wait_for(copying.wait(), 3)
            fold_release.set()
            await agent._session_state_owner.submit_internal(lambda: None)
            assert not turn.done()
            copy_release.set()
            await asyncio.wait_for(moving, 8)
            assert (await asyncio.wait_for(turn, 5)).ok
            assert agent.session_store is not source
            with pytest.raises(RuntimeError, match="closed"):
                await source.run(source.get_events, "alice")
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert sum(row["type"] == "tool_feedback_admitted" for row in rows) == 1
            messages = agent.controller.conversation.get_messages()
            assert (
                sum(
                    bool(message.metadata.get("feedback_admission_origin"))
                    for message in messages
                )
                == 1
            )
        finally:
            tool_release.set()
            provider_release.set()
            fold_release.set()
            copy_release.set()
            await asyncio.gather(
                turn, *([moving] if moving else []), return_exceptions=True
            )


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
        awaiting = EventEnvelope(
            event=create_user_input_event("awaited remains queued"), future=future
        )
        agent._event_inbox.put(EventEnvelope(event=event))
        agent._event_inbox.put(awaiting)
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
                if "_fold_bound_inputs.<locals>.commit" in callback.__qualname__:
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
        assert not future.done()
        assert agent._event_inbox.drain_all() == [awaiting]
        agent._event_inbox.put(awaiting)
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
                "_fold_bound_inputs.<locals>.commit" in callback.__qualname__
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
                "_fold_bound_inputs.<locals>.commit" in callback.__qualname__
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
