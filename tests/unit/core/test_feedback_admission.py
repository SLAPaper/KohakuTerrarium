"""Feedback persistence gates the exact text continuation on its source binding."""

import asyncio
import threading
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.errors import ConflictError, HistoryCompatibilityError
from kohakuterrarium.core import controller as controller_module, feedback_admission
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.history_compatibility import _native_groups
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.core.events import (
    EventType,
    TriggerEvent,
    create_tool_complete_event,
)
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium import session_coord
from kohakuterrarium.session.compact_compile import compile_coverage
from kohakuterrarium.session.resume_replay import replay_resume_history


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("compound", [False, True])
async def test_two_armed_feedback_occurrences_keep_independent_proofs(
    tmp_path, reverse, compound
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["seed"]),
            io="headless",
            start=True,
        )
        await creature.run("seed")
        agent, store = creature.agent, creature.agent.session_store
        events = [
            create_tool_complete_event(job, "same result") for job in ("job-a", "job-b")
        ]
        for event in events:

            def register():
                context = capture_tool_context(agent, event.job_id, "direct")
                agent._direct_job_meta[event.job_id] = {"completion_context": context}
                if not hasattr(agent, "_job_terminal_claims"):
                    agent._job_terminal_claims = {}
                agent._job_terminal_claims[event.job_id] = {
                    "kind": "tool",
                    "admitted": True,
                }
                return context

            context = await agent._session_state_owner.submit_internal(register)
            for kind in ("tool_call", "tool_result"):
                await store.run(
                    store.append_event,
                    "alice",
                    kind,
                    {
                        "job_id": event.job_id,
                        "call_id": event.job_id,
                        "origin": context["origin"],
                        "output": "same result",
                    },
                )
            await agent._session_state_owner.submit_internal(
                lambda: feedback_admission.arm_feedback(agent, event, [event.job_id])
            )
        await agent._session_state_owner.submit_internal(
            lambda: feedback_admission.arm_feedback(
                agent, events[0], [events[0].job_id]
            )
        )
        with pytest.raises(ConflictError, match="different evidence"):
            await agent._session_state_owner.submit_internal(
                lambda: feedback_admission.arm_feedback(
                    agent, events[0], [events[1].job_id]
                )
            )
        await agent._session_state_owner.submit_internal(
            lambda: feedback_admission.clear_feedback(agent, object())
        )
        clone = create_tool_complete_event("job-a", "same result")
        assert (
            await feedback_admission.prepare_feedback(
                agent.controller, [clone], "untrusted clone", False
            )
            is None
        )
        ordered = list(reversed(events)) if reverse else events
        if compound:
            await controller_module.capture_turn_messages(
                agent.controller,
                ordered,
                "\n\n".join(event.content for event in ordered),
                False,
            )
        else:
            for event in ordered:
                await controller_module.capture_turn_messages(
                    agent.controller, [event], event.content, False
                )
        rows = await store.run(store.get_events, "alice")
        admitted = [row for row in rows if row["type"] == "tool_feedback_admitted"]
        assert len(admitted) == (1 if compound else 2)
        assert sum(len(row["deliveries"]) for row in admitted) == 2
        assert sum(len(row["source_event_origins"]) for row in admitted) == 4
        proofs = [
            message.metadata["feedback_admission_origin"]
            for message in agent.controller.conversation.get_messages()
            if message.metadata.get("feedback_admission_origin")
        ]
        assert len(proofs) == (1 if compound else 2)
        assert len({tuple(proof) for proof in proofs}) == len(proofs)
        assert getattr(agent, "_text_feedback_admission", None) is None
        for event in events:
            assert (
                await feedback_admission.prepare_feedback(
                    agent.controller, [event], event.content, False
                )
                is None
            )


@pytest.mark.parametrize(
    "fold",
    [
        False,
        True,
        "mixed",
        "reverse",
        "multiple",
        "modal",
        "trigger",
        "batch",
        "batch-reverse",
    ],
)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("child", [False, True])
async def test_published_background_feedback_has_exact_live_and_replay_receipts(
    tmp_path, native, child, fold, monkeypatch
):
    release, continued = asyncio.Event(), asyncio.Event()
    mixed = isinstance(fold, str)
    provider_waiting, provider_release, queued = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    class Work(BaseTool):
        tool_name = "work"
        description = "Background work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            await release.wait()
            return ToolResult(output="background receipt result")

    class Parent(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            self.last_tool_calls = []
            if fold and self.call_count == 1:
                provider_waiting.set()
                await provider_release.wait()
            if self.call_count:
                continued.set()
            if native and self.call_count == 0:
                self.call_count += 1
                self.last_tool_calls = [
                    NativeToolCall(
                        "provider", "work", '{"task":"work", "run_in_background":true}'
                    )
                ]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native" if native else "bracket"),
            llm=Parent(["[/work]@@run_in_background=true\n{}[work/]", "received"]),
            tools=[] if child else [Work()],
            io="headless",
            start=True,
        )
        agent, store = creature.agent, creature.agent.session_store
        if child:

            class Child(ScriptedLLM):
                async def chat(self, messages, **kwargs):
                    await release.wait()
                    async for chunk in super().chat(messages, **kwargs):
                        yield chunk

            config = SubAgentConfig(
                name="work", description="held child", tools=[], max_turns=1
            )
            agent.subagent_manager.register(config)
            agent.registry.register_subagent("work", config)
            agent.subagent_manager.llm = Child(["background receipt result"])
        capture = controller_module.capture_turn_messages

        async def reject_clone(controller, events, content, skip_empty):
            if events and all(
                event.type == EventType.TOOL_COMPLETE and event.content
                for event in events
            ):
                clones = deepcopy(events)
                await agent._session_state_owner.submit_internal(
                    lambda: feedback_admission.arm_background_feedback(agent, clones)
                )
                pending = agent._text_feedback_admission
                assert all(id(clone) not in pending for clone in clones)
                assert all(id(event) in pending for event in events)
                assert (
                    await feedback_admission.prepare_feedback(
                        controller, clones, content, False
                    )
                    is None
                )
            return await capture(controller, events, content, skip_empty)

        monkeypatch.setattr(controller_module, "capture_turn_messages", reject_clone)
        await creature.run("start work")
        if fold in ("batch", "batch-reverse"):
            agent.pause()
            put = agent._event_inbox.put

            def observe_batch(envelope):
                result = put(envelope)
                if envelope.event.type == EventType.TOOL_COMPLETE:
                    queued.set()
                return result

            monkeypatch.setattr(agent._event_inbox, "put", observe_batch)
            if fold == "batch":
                assert await agent.inject_input("folded human") is False
            release.set()
            await asyncio.wait_for(queued.wait(), 4)
            if fold == "batch-reverse":
                assert await agent.inject_input("folded human") is False
            provider_release.set()
            agent.resume()
            await asyncio.wait_for(continued.wait(), 4)
        elif fold:
            put = agent._event_inbox.put

            def observe_put(envelope):
                result = put(envelope)
                if envelope.event.type == EventType.TOOL_COMPLETE:
                    queued.set()
                return result

            monkeypatch.setattr(agent._event_inbox, "put", observe_put)
            turn = asyncio.create_task(creature.run("independent work"))
            try:
                await asyncio.wait_for(provider_waiting.wait(), 3)
                if mixed and fold != "reverse":
                    if fold == "trigger":
                        assert (
                            await agent.inject_event(
                                TriggerEvent(type="trigger", content="folded human")
                            )
                            is False
                        )
                    else:
                        content = (
                            [{"type": "text", "text": "folded human"}]
                            if fold == "modal"
                            else "folded human"
                        )
                        assert await agent.inject_input(content) is False
                        if fold == "multiple":
                            assert (
                                await agent.inject_input("second folded human") is False
                            )
                release.set()
                await asyncio.wait_for(queued.wait(), 4)
                if fold == "reverse":
                    assert await agent.inject_input("folded human") is False
                provider_release.set()
                assert (await asyncio.wait_for(turn, 5)).ok
            finally:
                release.set()
                provider_release.set()
                await asyncio.gather(turn, return_exceptions=True)
        else:
            release.set()
            await asyncio.wait_for(continued.wait(), 4)
        async with agent._processing_lock:
            messages = [
                message
                for message in agent.controller.conversation.get_messages()
                if message.role == "user"
                and "background receipt result" in str(message.content)
            ]
            assert len(messages) == 1
            assert messages[0].metadata.get("feedback_admission_origin")
            if mixed:
                assert "folded human" in str(messages[0].content)
                assert messages[0].metadata.get("kind") != "tool_results"
                assert not any(
                    key in messages[0].metadata
                    for key in ("event_id", "user_pair_id", "user_pair_origin")
                )
            detached = {**messages[0].to_dict(), "metadata": messages[0].metadata}
            coverage = await store.run(compile_coverage, store, "alice", [detached], [])
            assert coverage["coverage_mode"] == "exact"
            rows = await store.run(store.get_events, "alice")
            assert sum(row["type"] == "tool_feedback_admitted" for row in rows) == 1
            if mixed:
                injected = [
                    row
                    for row in rows
                    if row["type"] == "user_input_injected"
                    and row.get("input_type") in ("user_input", "trigger")
                ]
                assert len(injected) == (
                    0 if fold == "batch" else 2 if fold == "multiple" else 1
                )
                assert (
                    sum(
                        message.get("content") == "folded human"
                        for message in replay_resume_history(rows)
                    )
                    == 0
                )
                damaged = deepcopy(rows)
                admission = next(
                    row for row in damaged if row["type"] == "tool_feedback_admitted"
                )
                input_proof = next(
                    proof
                    for proof in admission["deliveries"]
                    if proof.get("mode") in ("injected_input", "user_pair")
                )
                admission["source_event_origins"].remove(input_proof["event_origin"])
                with pytest.raises(HistoryCompatibilityError):
                    replay_resume_history(damaged)
            restored = replay_resume_history(rows)
            if native:
                _native_groups(restored, allow_pending_tail=False)
                assert (
                    sum(
                        message.get("metadata", {})
                        .get("tool_delivery", {})
                        .get("phase")
                        == "ack"
                        for message in restored
                    )
                    == 1
                )
            compact = {
                "event_id": max(row["event_id"] for row in rows) + 1,
                "type": "compact_complete",
                "summary": "background feedback summary",
                **coverage,
            }
            compacted = replay_resume_history([*rows, compact])
            assert (
                sum(
                    "background feedback summary" in str(message.get("content"))
                    for message in compacted
                )
                == 1
            )
            assert not any(
                message.get("role") == "user"
                and "background receipt result" in str(message.get("content"))
                for message in compacted
            )
            if native:
                _native_groups(compacted, allow_pending_tail=False)
                assert (
                    sum(
                        message.get("metadata", {})
                        .get("tool_delivery", {})
                        .get("phase")
                        == "ack"
                        for message in compacted
                    )
                    == 1
                )
            assert (
                sum(
                    message.get("role") == "user"
                    and "background receipt result" in str(message.get("content"))
                    for message in restored
                )
                == 1
            )


async def test_text_feedback_waits_for_copy_before_persisting_and_installing(
    tmp_path, monkeypatch
):
    feedback_ready, allow_feedback, copying = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    release = threading.Event()
    loop = asyncio.get_running_loop()
    capture_messages = controller_module.capture_turn_messages

    async def paused(controller, events, content, skip_empty):
        if events and all(event.type == EventType.TOOL_COMPLETE for event in events):
            feedback_ready.set()
            await allow_feedback.wait()
        return await capture_messages(controller, events, content, skip_empty)

    class Work(BaseTool):
        tool_name = "work"
        description = "Work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="confirmed result")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM(["[/work]{}[work/]", "done"]),
            tools=[Work()],
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        agent, source = alice.agent, alice.agent.session_store
        monkeypatch.setattr(controller_module, "capture_turn_messages", paused)
        capture = session_coord.capture_runtime_transfer
        first = True

        def held(*args):
            nonlocal first
            result = capture(*args)
            if first and args[0] is source:
                first = False
                loop.call_soon_threadsafe(copying.set)
                assert release.wait(6)
            return result

        monkeypatch.setattr(session_coord, "capture_runtime_transfer", held)
        turn = asyncio.create_task(alice.run("question", raise_on_error=False))
        moving = None
        try:
            await asyncio.wait_for(feedback_ready.wait(), 3)
            moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
            await asyncio.wait_for(copying.wait(), 3)
            allow_feedback.set()
            await agent._session_state_owner.submit_internal(lambda: None)
            release.set()
            await asyncio.wait_for(moving, 8)
            outcome = await asyncio.wait_for(turn, 5)
            assert outcome.ok and outcome.text == "done"
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert sum(row["type"] == "tool_feedback_admitted" for row in rows) == 1
            assert (
                sum(
                    bool(message.metadata.get("feedback_admission_origin"))
                    for message in agent.controller.conversation.get_messages()
                )
                == 1
            )
        finally:
            allow_feedback.set()
            release.set()
            await asyncio.gather(
                turn, *([moving] if moving else []), return_exceptions=True
            )


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "write_failure",
        "write_then_fail",
        "conversation",
        "branch",
        "view",
        "namespace",
        "scope_done",
    ],
)
async def test_feedback_worker_barrier_blocks_continuation_on_changed_source(
    tmp_path, monkeypatch, mode, cancelled
):
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    persist = feedback_admission.persist_feedback
    llm = ScriptedLLM(["[/work]{}[work/]", "done"])

    def held(*args):
        assert threading.current_thread() is not threading.main_thread()
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(8)
        if mode == "write_failure":
            raise OSError("feedback append failed")
        result = persist(*args)
        if mode == "write_then_fail":
            raise OSError("feedback written but unconfirmed")
        return result

    class Work(BaseTool):
        tool_name = "work"
        description = "Work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="confirmed result")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=llm,
            tools=[Work()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        monkeypatch.setattr(feedback_admission, "persist_feedback", held)
        capture, callers = controller_module.capture_turn_messages, []

        async def observed(controller, events, content, skip_empty):
            if events and all(
                event.type == EventType.TOOL_COMPLETE for event in events
            ):
                callers.append(asyncio.current_task())
            return await capture(controller, events, content, skip_empty)

        monkeypatch.setattr(controller_module, "capture_turn_messages", observed)
        turn = asyncio.create_task(creature.run("question", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert llm.call_count == 1
            if mode == "conversation":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(agent.controller, "conversation", Conversation())
                )
            elif mode == "branch":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(agent, "_branch_id", agent._branch_id + 1)
                )
            elif mode == "view":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(agent, "_branch_view", {1: 99})
                )
            elif mode == "namespace":
                sink = feedback_admission._canonical_output(agent.controller)
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(sink, "_event_key_prefix", "other")
                )
            elif mode == "scope_done":
                await agent._session_state_owner.submit_internal(
                    lambda: agent._turn_completion.set_result(None)
                )
            if cancelled:
                callers[0].cancel()
                await agent._session_state_owner.submit_internal(lambda: None)
                assert not callers[0].done()
            release.set()
            result = await asyncio.wait_for(turn, 8)
            installed = mode == "success" and not cancelled
            assert result.ok == installed
            assert llm.call_count == (2 if installed else 1)
            if not installed:
                with pytest.raises(ConflictError, match="recovery"):
                    require_model_mutation_available(agent)
            messages = agent.controller.conversation.get_messages()
            admitted = [
                message
                for message in messages
                if message.metadata.get("feedback_admission_origin")
            ]
            assert len(admitted) == (1 if installed else 0)
            assert getattr(agent, "_text_feedback_admission", None) is None
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert len(
                [row for row in rows if row["type"] == "tool_feedback_admitted"]
            ) == (0 if mode == "write_failure" else 1)
        finally:
            release.set()
            await asyncio.gather(turn, return_exceptions=True)
