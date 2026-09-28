"""Text dispatch acknowledgements compose without claiming unfinished job results."""

import asyncio
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.output.base import BaseOutputModule
from kohakuterrarium.session.compact_compile import compile_coverage
from kohakuterrarium.session.compact_coverage import coordinate
from kohakuterrarium.session.resume_replay import replay_resume_history
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.core.output_feedback import take_output_proofs


@pytest.mark.parametrize("with_tool", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_named_output_and_direct_result_have_confirmed_constituents(
    tmp_path, fails, with_tool, monkeypatch
):
    writes = []

    class Target(BaseOutputModule):
        async def write(self, content):
            writes.append(content)
            if fails:
                raise OSError("destination unavailable")

        async def write_stream(self, content):
            pass

    class Echo(BaseTool):
        tool_name = "echo"
        description = "Immediate output"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="direct output")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM(
                [
                    "[/output_notice]delivered payload[output_notice/]"
                    + ("[/echo]{}[echo/]" if with_tool else ""),
                    "observed delivery",
                ]
            ),
            tools=[Echo()],
            io="headless",
            start=True,
        )
        agent, store = creature.agent, creature.agent.session_store
        agent.output_router.named_outputs["notice"] = Target()
        agent.controller.config.known_outputs.add("notice")
        get_feedback = agent.output_router.get_output_feedback

        def probe_clones():
            completions = agent.output_router.completed_outputs
            if completions:
                assert take_output_proofs(agent, deepcopy(completions)) is None
            return get_feedback()

        monkeypatch.setattr(agent.output_router, "get_output_feedback", probe_clones)
        assert (await creature.run("send and compute", timeout=5)).ok
        assert writes == ["delivered payload"]
        message = next(
            message
            for message in agent.controller.conversation.get_messages()
            if message.role == "user" and "Outputs Sent" in str(message.content)
        )
        assert message.metadata.get("feedback_admission_origin")
        assert ("FAILED" in message.content) == fails
        rows = await store.run(store.get_events, "alice")
        deliveries = [row for row in rows if row["type"] == "output_delivery"]
        assert len(deliveries) == 1 and deliveries[0]["success"] is (not fails)
        coverage = await store.run(
            compile_coverage,
            store,
            "alice",
            [{**message.to_dict(), "metadata": deepcopy(message.metadata)}],
            [],
        )
        assert coverage["coverage_mode"] == "exact"
        assert list(coordinate(deliveries[0])) in coverage["replaced_event_origins"]
        restored = replay_resume_history(rows)
        assert sum(item.get("content") == message.content for item in restored) == 1
        damaged = deepcopy(rows)
        next(row for row in damaged if row["type"] == "output_delivery")[
            "content"
        ] = "different output"
        with pytest.raises(HistoryCompatibilityError):
            replay_resume_history(damaged)
        compact = {
            "event_id": max(row["event_id"] for row in rows) + 1,
            "type": "compact_complete",
            "summary": "output feedback summary",
            **coverage,
        }
        restored = replay_resume_history([*rows, compact])
        assert (
            sum(
                "output feedback summary" in str(item.get("content"))
                for item in restored
            )
            == 1
        )
        assert not any("Outputs Sent" in str(item.get("content")) for item in restored)


@pytest.mark.parametrize("promote", [False, True])
@pytest.mark.parametrize("background_child", [False, True])
@pytest.mark.parametrize("direct_child", [False, True])
async def test_text_ack_and_direct_result_are_proven_before_background_finishes(
    tmp_path, background_child, direct_child, promote
):
    release, started = asyncio.Event(), asyncio.Event()

    class Work(BaseTool):
        tool_name = "work"
        description = "Pending background work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            started.set()
            await release.wait()
            return ToolResult(output="late background output")

    class Echo(BaseTool):
        tool_name = "echo"
        description = "Immediate output"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="immediate output")

    direct_body = "@@run_in_background=false\nimmediate task"
    pending_body = (
        "@@run_in_background=false\npending task"
        if promote
        else "@@run_in_background=true\npending task"
    )
    script = "[/work]" + pending_body + "[work/][/echo]" + direct_body + "[echo/]"
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM([script, "received"]),
            tools=([Work()] if not background_child else [])
            + ([Echo()] if not direct_child else []),
            io="headless",
            start=True,
        )
        agent, store = creature.agent, creature.agent.session_store

        class Child(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                if any("pending task" in str(message) for message in messages):
                    started.set()
                    await release.wait()
                    yield "late background output"
                else:
                    yield "immediate output"

        agent.subagent_manager.llm = Child([])
        for name, enabled in (("work", background_child), ("echo", direct_child)):
            if enabled:
                config = SubAgentConfig(
                    name=name, description=name, tools=[], max_turns=1
                )
                agent.subagent_manager.register(config)
                agent.registry.register_subagent(name, config)
        pending = asyncio.create_task(creature.run("dispatch both", timeout=5))
        try:
            if promote:
                await asyncio.wait_for(started.wait(), 3)
                job = next(
                    job_id
                    for job_id in agent._active_handles
                    if job_id.startswith("agent_work" if background_child else "work")
                )
                assert agent._promote_handle(job)
            result = await pending
            assert result.ok and result.text == "received"
            message = next(
                message
                for message in agent.controller.conversation.get_messages()
                if message.role == "user" and "immediate output" in str(message.content)
            )
            assert "background" in message.content.lower()
            assert message.metadata.get("feedback_admission_origin")
            coverage = await store.run(
                compile_coverage,
                store,
                "alice",
                [{**message.to_dict(), "metadata": deepcopy(message.metadata)}],
                [],
            )
            assert coverage["coverage_mode"] == "exact"
            rows = await store.run(store.get_events, "alice")
            background_call = next(
                row
                for row in rows
                if row["type"] in ("tool_call", "subagent_call")
                and row["name"] == "work"
            )
            assert (
                list(coordinate(background_call)) in coverage["replaced_event_origins"]
            ), str(
                [
                    (
                        row.get("event_id"),
                        row.get("type"),
                        row.get("name"),
                        row.get("deliveries"),
                    )
                    for row in rows
                    if row.get("type")
                    in ("tool_call", "tool_result", "tool_feedback_admitted")
                ]
            )
            assert not any(
                row["type"] in ("tool_result", "subagent_result")
                and row.get("job_id") == background_call["job_id"]
                for row in rows
            )
            restored = replay_resume_history(rows)
            assert sum(item.get("content") == message.content for item in restored) == 1
            incomplete = deepcopy(rows)
            admission = next(
                row for row in incomplete if row["type"] == "tool_feedback_admitted"
            )
            admission["source_event_origins"].remove(list(coordinate(background_call)))
            with pytest.raises(HistoryCompatibilityError):
                replay_resume_history(incomplete)
            damaged = deepcopy(rows)
            call = next(
                row
                for row in damaged
                if row.get("job_id") == background_call["job_id"]
                and row["type"] in ("tool_call", "subagent_call")
            )
            call["origin"]["delivery_id"] = "foreign delivery"
            with pytest.raises(HistoryCompatibilityError):
                replay_resume_history(damaged)
            agent.pause()
            release.set()
            if background_child:
                await agent.subagent_manager.wait_all(timeout=3)
            else:
                await agent.executor.wait_all(timeout=3)
            await agent.output_router.flush()
            rows = await store.run(store.get_events, "alice")
            restored = replay_resume_history(rows)
            assert sum(item.get("content") == message.content for item in restored) == 1
            assert (
                sum(
                    "late background output" in str(item.get("content"))
                    for item in restored
                )
                == 1
            )
            late = [
                item
                for item in restored
                if "late background output" in str(item.get("content"))
            ]
            late_coverage = await store.run(compile_coverage, store, "alice", late, [])
            assert late_coverage["coverage_mode"] == "exact"
            assert (
                list(coordinate(background_call))
                not in late_coverage["replaced_event_origins"]
            )
            compact = {
                "event_id": max(row["event_id"] for row in rows) + 1,
                "type": "compact_complete",
                "summary": "dispatch feedback summary",
                **coverage,
            }
            recovered = replay_resume_history([*rows, compact])
            assert (
                sum(
                    "dispatch feedback summary" in str(item.get("content"))
                    for item in recovered
                )
                == 1
            )
            assert (
                sum(
                    "late background output" in str(item.get("content"))
                    for item in recovered
                )
                == 1
            )
        finally:
            agent.pause()
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
